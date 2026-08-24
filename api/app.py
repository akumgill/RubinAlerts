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
from .selection import SelectionStore, compute_persistence, MAX_PAYLOAD_BYTES
from .observations import ObservationStore, normalize_night

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

from .resolver import resolve_name

svc = TargetQueueService(PROGRAMS, ALLOCATIONS, db_path=DB_PATH,
                         resolver=resolve_name)

# Nightly SN Ia selection results (separate table in the same DB file; not
# part of the dashboard cache).
selection_store = SelectionStore(db_path=DB_PATH)

# Ingested observations (what was actually shot on sky) + their per-program
# time charges. Same DB file; separate tables (see api/observations.py).
observation_store = ObservationStore(db_path=DB_PATH)

# Programs allowed to READ the selection view (uploads only need a valid
# identity — the pipeline posts with its bearer key).
SELECTION_PROGRAMS = frozenset(
    p.strip() for p in
    os.environ.get("SELECTION_PROGRAMS", "CfA-Stubbs").split(",") if p.strip())

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
# ETC suggestion (stamped #2): exposure from anticipated magnitude via the
# LLAMAS S/N ETC. This is a pre-filled EDITABLE suggestion for the add-target
# form — the submitter still owns (and must state) the exposure; nothing is
# ever silently auto-sized.
# ---------------------------------------------------------------------------
@app.get("/v1/etc")
def etc_suggest(request: Request, mag: float, authorization: str = Header(None)):
    """Suggested LLAMAS exposure for an anticipated magnitude, expressed as
    the canonical cosmic-ray-rejection triplet: 3 equal sub-exposures rounded
    to the nearest 10 s. Returns {minutes, n_exposures, exposure_seconds,
    snr, n_bin, extrapolated}."""
    _identity(request, authorization)
    try:
        from core.snr_etc import (snr_exposure_minutes, MIN_EXPOSURE_MINUTES,
                                  MAX_EXPOSURE_MIN, DEFAULT_TARGET_SNR,
                                  DEFAULT_N_BIN)
    except Exception as e:  # container without the ETC module/curve
        logger.warning("ETC unavailable: %s", e)
        raise HTTPException(503, "exposure calculator unavailable on this "
                                 "deployment — enter the exposure manually")
    import math
    if not math.isfinite(mag):
        raise HTTPException(422, "mag must be a finite magnitude")
    t, extrapolated = snr_exposure_minutes(mag)
    if not math.isfinite(t):
        raise HTTPException(422, "ETC could not size an exposure for this mag")
    minutes = float(min(MAX_EXPOSURE_MIN, max(MIN_EXPOSURE_MINUTES, t)))
    # canonical CR-median protocol: 3 equal sub-exposures, each a multiple of 10 s
    sub_s = max(10.0, round(minutes * 60.0 / 3.0 / 10.0) * 10.0)
    return {"minutes": round(3 * sub_s / 60.0, 2), "n_exposures": 3,
            "exposure_seconds": sub_s, "snr": DEFAULT_TARGET_SNR,
            "n_bin": DEFAULT_N_BIN, "extrapolated": bool(extrapolated)}


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
    dash = _guard(lambda: dashboard_data(svc, date, instrument,
                                         caller_program=program))
    # burndown: fold ingested observed time into `used` OUTSIDE the cache
    # (shallow copy so the cached payload itself is never mutated)
    dash = dict(dash)
    dash["allocations"] = _fold_observed_time(dash.get("allocations"))
    return dash


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
# SN Ia target selection: nightly ranked candidates from the alert pipeline.
# Ingest (POST) needs any valid identity — the pipeline uploads with its bearer
# key. The read view is limited to the programs in SELECTION_PROGRAMS.
# ---------------------------------------------------------------------------
@app.post("/v1/selection/nights")
def upload_selection_night(request: Request, payload: dict = Body(...),
                           authorization: str = Header(None)):
    _identity(request, authorization)
    candidates = payload.get("candidates") or []
    summary = payload.get("summary") or {}
    if len(json.dumps({"summary": summary, "candidates": candidates})) > MAX_PAYLOAD_BYTES:
        raise HTTPException(413, f"payload too large (max {MAX_PAYLOAD_BYTES} bytes)")
    return _guard(lambda: selection_store.upsert_night(
        str(payload.get("night_stamp") or ""), payload.get("mjd"),
        summary, candidates))


@app.get("/v1/selection")
def selection(request: Request, limit_nights: int = 10,
              authorization: str = Header(None)):
    _, program = _identity(request, authorization)
    if program not in SELECTION_PROGRAMS:
        raise HTTPException(
            403, f"the SN Ia selection view is limited to "
                 f"{', '.join(sorted(SELECTION_PROGRAMS))}; "
                 f"your program ({program}) does not have access")
    nights = _guard(lambda: selection_store.fetch_nights(limit_nights))
    return {"nights": nights, "persistence": compute_persistence(nights)}


# ---------------------------------------------------------------------------
# Observing-plan bundle for an operator batch (item G): the top-of-queue picks
# rendered as the three files an observer actually loads. Batches of 4-6 only
# (Chris's protocol) — never full nights; exposures as CR-median triplets.
# ---------------------------------------------------------------------------
def _obsplan_row(t) -> dict:
    """Queue target -> serializer row. Exposure: the target's own triplet
    spec when set, else the total split into 3 equal subs rounded to 10 s
    (the canonical CR protocol), else the 45-min fallback as 3 x 900 s."""
    import math as _math
    if t.n_exposures and _math.isfinite(t.exposure_seconds):
        n_exp, exp_sec = int(t.n_exposures), float(t.exposure_seconds)
    else:
        total_min = (t.exposure_minutes
                     if _math.isfinite(t.exposure_minutes) else 45.0)
        n_exp = 3
        exp_sec = max(10.0, round(total_min * 60.0 / 3.0 / 10.0) * 10.0)
    return {"name": t.name or f"{t.program}-{t.id}",
            "ra": t.canonical_ra, "dec": t.canonical_dec,
            "mag": t.mag if _math.isfinite(t.mag) else None,
            "priority": t.priority, "instrument": t.instrument,
            "airmass_min": t.airmass_min, "airmass_max": t.airmass_max,
            "n_exp": n_exp, "exp_sec": exp_sec, "notes": t.notes or ""}


@app.post("/v1/obsplan")
def obsplan(request: Request, payload: dict = Body(...),
            authorization: str = Header(None)):
    """Render 1-6 queue targets as an observing bundle: {catalog_cat,
    plan_txt, instrument_macro, order, date}. Ordered by tonight's
    observability (window start); input order when planning is unavailable."""
    import math
    from datetime import datetime, timezone
    _identity(request, authorization)
    ids = payload.get("target_ids") or []
    instrument = str(payload.get("instrument") or "LLAMAS").upper()
    if not isinstance(ids, list) or not ids:
        raise HTTPException(422, "target_ids must be a non-empty list")
    if len(ids) > 6:
        raise HTTPException(
            422, f"{len(ids)} targets requested — observing plans are "
                 "generated in batches of 4-6 (never full nights); trim the "
                 "selection")
    by_id = {t.id: t for t in svc._targets}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise HTTPException(422, f"unknown target ids: {missing}")
    rows = [_obsplan_row(by_id[i]) for i in ids]
    date = payload.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # tonight's ordering: observability-window start (the planner's geometry);
    # fall back to the operator's input order if the solve fails
    try:
        from orchestrator.planner import calculate_twilight, compute_observability
        from orchestrator.models import Target as OTarget
        evening, morning = calculate_twilight(date)
        ots = [OTarget(name=r["name"], ra_deg=r["ra"], dec_deg=r["dec"],
                       exposure_minutes=r["n_exp"] * r["exp_sec"] / 60.0,
                       airmass_min=r.get("airmass_min", float("nan")),
                       airmass_max=r.get("airmass_max", float("nan")))
               for r in rows]
        not_obs: list = []
        obs = compute_observability(ots, evening, morning, dropped=not_obs)
        start_mjd = {t.name: t.window_start.mjd for t in obs
                     if t.window_start is not None}
        rows.sort(key=lambda r: start_mjd.get(r["name"], float("inf")))
        # Stamp WHEN to shoot each row into the notes column. Three pseudo-
        # targets of one standard are three identical pointings on the sheet —
        # without the window the observer cannot tell which to take at 02:00.
        # Goes in notes (free text) rather than a new column: the 12-column
        # LDSS_ObsPlan_Generator convention stays byte-compatible.
        _win = {t.name: t for t in obs}
        _dropped = {t.name: reason for t, reason in not_obs}
        for r in rows:
            t = _win.get(r["name"])
            if t is None or t.window_start is None or t.window_end is None:
                # The operator put it in the batch and it cannot be shot
                # tonight — say so on the sheet rather than leaving a row that
                # looks identical to its schedulable siblings.
                reason = _dropped.get(r["name"])
                if reason:
                    stamp = f"NOT OBSERVABLE — {reason}"
                    r["notes"] = (f"{r['notes']}; {stamp}" if r["notes"]
                                  else stamp)
                continue
            span = (f"{t.window_start.datetime.strftime('%H:%M')}-"
                    f"{t.window_end.datetime.strftime('%H:%M')} UT")
            lo, hi = r.get("airmass_min"), r.get("airmass_max")
            if lo is not None and hi is not None and math.isfinite(lo) \
                    and math.isfinite(hi):
                stamp = f"observe {span} (airmass {lo:.1f}-{hi:.1f})"
            elif math.isfinite(t.min_airmass):
                stamp = f"observe {span} (min airmass {t.min_airmass:.2f})"
            else:
                stamp = f"observe {span}"
            r["notes"] = f"{r['notes']}; {stamp}" if r["notes"] else stamp
    except Exception as e:
        logger.warning("obsplan: observability ordering failed (%s); "
                       "keeping input order", e)
    from orchestrator.obsfiles import tcs_catalog, plan_sheet, llamas_macro
    return {"date": date, "instrument": instrument,
            "order": [r["name"] for r in rows],
            "catalog_cat": tcs_catalog(rows),
            "plan_txt": plan_sheet(rows, date, instrument),
            "instrument_macro": llamas_macro(rows)}


# ---------------------------------------------------------------------------
# Observation ingestion + observed repository (item F). The FITS source is
# mocked for now: scripts/ingest_fits_night.py adapts headers into canonical
# records and POSTs them here; the server owns association + accounting.
# ---------------------------------------------------------------------------
@app.post("/v1/observations")
def ingest_observations(request: Request, payload: dict = Body(...),
                        authorization: str = Header(None)):
    """Ingest a night's canonical observation records. The server associates
    each record to the queue (pointing within 1 arcmin, airmass-bin
    disambiguation for standards, name fallback) and records per-program time
    charges (even split for shared coordinates; idempotent by filename)."""
    _identity(request, authorization)
    records = payload.get("observations") or []
    if not isinstance(records, list) or not records:
        raise HTTPException(422, "body must be {observations: [record, ...]}")
    if len(records) > 1000:
        raise HTTPException(422, "too many records in one batch (max 1000)")
    results = observation_store.ingest(svc._targets, records)
    n_assoc = sum(1 for r in results
                  if r.get("assoc_method") in ("pointing", "name"))
    return {"results": results, "n_records": len(results),
            "n_associated": n_assoc,
            "n_unassociated": sum(1 for r in results
                                  if r.get("assoc_method") == "unassociated")}


@app.get("/v1/observations")
def observations(request: Request, night: str = None,
                 authorization: str = Header(None)):
    """With ?night=YYYY-MM-DD (or utYYYYMMDD): that night's observation rows.
    Without: the compact observed-pointings list + per-target latest-night
    map, for 'observed' badges on the dashboard/selection pages."""
    _identity(request, authorization)
    if night:
        return {"night_stamp": normalize_night(night),
                "observations": observation_store.night_rows(night)}
    return {"observed_coords": observation_store.observed_coords(),
            "observed_targets": observation_store.observed_target_ids()}


def _fold_observed_time(alloc: dict) -> dict:
    """Fold ingested observation charges into the allocations overview's
    'used' (the static YAML overview carries used=0 + tracked=False until
    real observations flow in). Runs per request — charges are cheap to read
    and must not be frozen into the dashboard cache."""
    used = observation_store.used_hours_by_program()
    if not alloc or not used:
        return alloc
    import copy
    alloc = copy.deepcopy(alloc)
    tracked = False
    for prog, by_inst in used.items():
        rows = (alloc.get("programs") or {}).get(prog)
        if not rows:
            logger.debug("observed time for %r matches no allocation row", prog)
            continue
        for inst, hours in by_inst.items():
            d = rows.get(inst)
            if not d:
                continue
            d["used"] = round(d.get("used", 0.0) + hours, 2)
            d["remaining"] = round(d["initial"] - d["used"], 2)
            tracked = True
    if tracked:
        alloc["tracked"] = True
    return alloc


# ---------------------------------------------------------------------------
# Static frontend (mounted last so it doesn't shadow the API routes). The web/
# directory is owned by the frontend build; we only guarantee it exists.
# ---------------------------------------------------------------------------
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
os.makedirs(_WEB_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
