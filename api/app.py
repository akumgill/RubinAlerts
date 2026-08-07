"""FastAPI wrapper over TargetQueueService — the deployment layer.

This is the only part that needs a web framework; the service core does not.
Run with:  uvicorn api.app:app --host 0.0.0.0 --port 8000

Two ways to authenticate, both resolving to a program:
  * a bearer API key (machine clients / the automated pipeline), or
  * a signed session cookie set by POST /login (browser users).
Reads accept either; writes stay scoped to the caller's own program.
"""
from __future__ import annotations

import json
import logging
import os

from fastapi import FastAPI, Header, HTTPException, Body, Form, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .service import TargetQueueService, AuthError, NotFound

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Group config: program -> {key, password}. From GROUPS_JSON (a JSON blob), or
# per-program env vars, else the built-in demo groups (UA + CfA/Villar) so a
# fresh deploy is immediately usable.
# ---------------------------------------------------------------------------
def _load_group_config() -> dict:
    raw = os.environ.get("GROUPS_JSON")
    if raw:
        try:
            return json.loads(raw)
        except Exception as e:
            logger.error("GROUPS_JSON is not valid JSON: %s", e)
            # fall through to the fail-closed / demo decision below
    # No valid GROUPS_JSON. Fall back to the built-in demo groups ONLY in dev
    # (no real session secret). In production — signalled by a real
    # SESSION_SECRET — refuse to serve the shared queue on publicly-known demo
    # credentials; fail closed so a forgotten env var can't silently expose it.
    secret = os.environ.get("SESSION_SECRET")
    if secret and secret != "dev-insecure-session-secret":
        raise RuntimeError(
            "GROUPS_JSON is unset or invalid but a production SESSION_SECRET is "
            "set — refusing to fall back to public demo credentials. Set "
            "GROUPS_JSON to the real program config.")
    from .seed import demo_group_config
    logger.warning("GROUPS_JSON unset; using built-in demo groups (dev only)")
    return demo_group_config()


GROUPS = _load_group_config()
PROGRAMS = {cfg["key"]: prog for prog, cfg in GROUPS.items()}         # key -> program
PROGRAM_TO_KEY = {prog: cfg["key"] for prog, cfg in GROUPS.items()}   # program -> key
PASSWORDS = {prog: cfg.get("password") for prog, cfg in GROUPS.items()}

DB_PATH = os.environ.get("DB_PATH", "./data/queue.db")
DATA_DIR = os.path.dirname(os.path.abspath(DB_PATH)) or "."

# Allocations: explicit path, else the demo allocations written into DATA_DIR.
ALLOCATIONS = os.environ.get("MAGNETS_ALLOCATIONS")
if not ALLOCATIONS:
    from .seed import ensure_demo_allocations
    ALLOCATIONS = ensure_demo_allocations(DATA_DIR)

SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-insecure-session-secret")
if SESSION_SECRET == "dev-insecure-session-secret":
    logger.warning("SESSION_SECRET is unset; using an insecure dev secret")

svc = TargetQueueService(PROGRAMS, ALLOCATIONS, db_path=DB_PATH)

# One-time demo seed on a fresh (empty) database.
if os.environ.get("SEED_DEMO") == "1" and not svc.has_targets():
    try:
        from .seed import seed_demo
        seed_demo(svc)
    except Exception as e:
        logger.exception("demo seed failed: %s", e)

# Default the landing view to the instrument the demo is seeded on (LDSS3 —
# the Villar list), so a first visitor sees a populated queue rather than an
# empty LLAMAS plan. Overridable via env.
DEFAULT_NIGHT = (os.environ.get("DEFAULT_DATE", "2026-08-13"),
                 os.environ.get("DEFAULT_INSTRUMENT", "LDSS3"))

# Warm the default-night dashboard cache in a background thread, so the first
# request after a (re)start hits a warm cache instead of paying the full
# scheduler + airmass compute. Daemon + non-fatal. Off in tests (WARM_CACHE=0).
if os.environ.get("WARM_CACHE", "1") == "1":
    import threading

    def _warm_cache():
        try:
            from .scheduler_bridge import dashboard_data
            dashboard_data(svc, DEFAULT_NIGHT[0], DEFAULT_NIGHT[1])
            logger.info("dashboard cache warmed for %s %s", *DEFAULT_NIGHT)
        except Exception as e:
            logger.warning("dashboard cache warm failed: %s", e)

    threading.Thread(target=_warm_cache, daemon=True, name="cache-warm").start()

app = FastAPI(title="MAGNETS Target-Submission API", version="0.3")
# Secure (HTTPS-only) session cookies in production (Render terminates TLS);
# off by default so plain-HTTP local dev / TestClient still round-trips the
# cookie. render.yaml sets SECURE_COOKIES=1.
_SECURE_COOKIES = os.environ.get("SECURE_COOKIES", "0") == "1"
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET,
                   same_site="lax", https_only=_SECURE_COOKIES)


@app.get("/healthz")
def healthz():
    """Unauthenticated liveness probe (Render health check)."""
    return {"ok": True}


@app.get("/v1/programs")
def programs():
    """Unauthenticated: the configured program names, so the login screen can
    offer every group (across both instruments) before anyone signs in."""
    return {"programs": sorted(GROUPS.keys())}


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def _identity(request: Request, authorization: str | None) -> tuple[str, str]:
    """Resolve the caller to (api_key, program) from a bearer key OR a session
    cookie. Raises 401 if neither is present/valid."""
    # 1. Bearer key
    if authorization and authorization.lower().startswith("bearer "):
        key = authorization.split(" ", 1)[1].strip()
        try:
            program = svc.program_for(key)
            return key, program
        except AuthError:
            raise HTTPException(401, "invalid API key")
    # 2. Session cookie
    program = request.session.get("program")
    if program and program in PROGRAM_TO_KEY:
        return PROGRAM_TO_KEY[program], program
    raise HTTPException(401, "authentication required (bearer key or login)")


def _guard(fn):
    try:
        return fn()
    except AuthError as e:
        raise HTTPException(401, str(e))
    except NotFound as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


# ---------------------------------------------------------------------------
# Browser auth
# ---------------------------------------------------------------------------
@app.post("/login")
def login(request: Request, program: str = Form(...), password: str = Form(...)):
    import secrets
    expected = PASSWORDS.get(program)
    if expected is None or not secrets.compare_digest(str(password), str(expected)):
        raise HTTPException(401, "invalid program or password")
    request.session["program"] = program
    return {"ok": True, "program": program}


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/v1/whoami")
def whoami(request: Request, authorization: str = Header(None)):
    key, program = _identity(request, authorization)
    return {"program": program}


# ---------------------------------------------------------------------------
# Target CRUD (writes scoped to the caller's program)
# ---------------------------------------------------------------------------
@app.post("/v1/targets")
def submit(request: Request, items: list[dict] = Body(...),
           authorization: str = Header(None)):
    key, _ = _identity(request, authorization)
    return _guard(lambda: svc.submit(key, items))


@app.get("/v1/targets")
def list_targets(request: Request, authorization: str = Header(None)):
    key, _ = _identity(request, authorization)
    return _guard(lambda: svc.list_targets(key))


@app.patch("/v1/targets/{target_id}")
def patch(request: Request, target_id: int, changes: dict = Body(...),
          authorization: str = Header(None)):
    key, _ = _identity(request, authorization)
    return _guard(lambda: svc.patch(key, target_id, changes))


@app.delete("/v1/targets/{target_id}")
def withdraw(request: Request, target_id: int, authorization: str = Header(None)):
    key, _ = _identity(request, authorization)
    return _guard(lambda: svc.withdraw(key, target_id))


# ---------------------------------------------------------------------------
# Reads (collaboration-wide; require a valid session or bearer)
# ---------------------------------------------------------------------------
@app.get("/v1/queue")
def queue(request: Request, authorization: str = Header(None)):
    _identity(request, authorization)
    return _guard(lambda: svc.queue_summary())


@app.get("/v1/plan/preview")
def plan_preview(request: Request, date: str, instrument: str = "LLAMAS",
                 moon: str = None, authorization: str = Header(None)):
    _identity(request, authorization)
    return _guard(lambda: svc.plan_preview(date, moon, instrument))


@app.get("/v1/dashboard")
def dashboard(request: Request, instrument: str = "LDSS3",
              date: str = "2026-08-13", authorization: str = Header(None)):
    """The full aggregate the web dashboard renders: plan + queue + per-target
    airmass tracks + grid + program metadata. Defaults to the seeded instrument
    (LDSS3) so the landing view is populated rather than empty."""
    _, program = _identity(request, authorization)
    from .scheduler_bridge import dashboard_data
    return _guard(lambda: dashboard_data(svc, date, instrument,
                                         caller_program=program))


@app.get("/v1/plan/export")
def export_plan(request: Request, instrument: str = "LDSS3",
                date: str = "2026-08-13", fmt: str = "catalog",
                authorization: str = Header(None)):
    """Export the plan the way an observer uses it: fmt=catalog (the instrument
    catalog the GUI loads), csv (an observing sheet), or text (a printable
    sheet)."""
    _, program = _identity(request, authorization)
    from .scheduler_bridge import dashboard_data
    from . import plan_export
    dash = dashboard_data(svc, date, instrument, caller_program=program)
    fmt = fmt.lower()
    if fmt == "catalog":
        body, ext, ctype = plan_export.catalog_text(dash), "cat", "text/plain"
    elif fmt == "csv":
        body, ext, ctype = plan_export.observing_csv(dash), "csv", "text/csv"
    elif fmt in ("text", "sheet"):
        body, ext, ctype = plan_export.observing_text(dash), "txt", "text/plain"
    else:
        raise HTTPException(400, "fmt must be catalog | csv | text")
    fname = f"MAGNETS_{instrument}_{date}.{ext}"
    return Response(content=body, media_type=ctype,
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


# ---------------------------------------------------------------------------
# Static frontend (mounted last so it doesn't shadow the API routes). The web/
# directory is owned by the frontend build; we only guarantee it exists.
# ---------------------------------------------------------------------------
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
os.makedirs(_WEB_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
