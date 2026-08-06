"""FastAPI wrapper over TargetQueueService — the deployment layer.

This is the only part that needs a web framework; the service core does not.
Run with:  uvicorn api.app:app --reload   (needs `pip install fastapi uvicorn`)

Auth is a bearer key that maps to a program (see the spec, section 6). For a
trusted ~dozen-user collaboration this is deliberately lightweight.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException, Body

from .service import TargetQueueService, AuthError, NotFound

# In a real deployment these come from config / a secrets file. Placeholder
# wiring so the module is runnable; replace with the collaboration's keys.
PROGRAMS = {
    os.environ.get("MAGNETS_IA_KEY", "key-ia"): "MAGNETS-Ia",
    os.environ.get("MAGNETS_EXOTIC_KEY", "key-exotic"): "MAGNETS-Exotic",
    os.environ.get("MAGNETS_OTHER_KEY", "key-other"): "MAGNETS-Other",
}
ALLOCATIONS = os.environ.get("MAGNETS_ALLOCATIONS", "ref/allocations_example.yaml")

svc = TargetQueueService(PROGRAMS, ALLOCATIONS)
app = FastAPI(title="MAGNETS Target-Submission API", version="0.3")


def _key(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    return authorization.split(" ", 1)[1].strip()


def _guard(fn):
    try:
        return fn()
    except AuthError as e:
        raise HTTPException(403, str(e))
    except NotFound as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/v1/targets")
def submit(items: list[dict] = Body(...), authorization: str = Header(None)):
    return _guard(lambda: svc.submit(_key(authorization), items))


@app.get("/v1/targets")
def list_targets(authorization: str = Header(None)):
    return _guard(lambda: svc.list_targets(_key(authorization)))


@app.patch("/v1/targets/{target_id}")
def patch(target_id: int, changes: dict = Body(...), authorization: str = Header(None)):
    return _guard(lambda: svc.patch(_key(authorization), target_id, changes))


@app.delete("/v1/targets/{target_id}")
def withdraw(target_id: int, authorization: str = Header(None)):
    return _guard(lambda: svc.withdraw(_key(authorization), target_id))


@app.get("/v1/queue")
def queue(authorization: str = Header(None)):
    return _guard(lambda: (svc.program_for(_key(authorization)), svc.queue_summary())[1])


@app.get("/v1/plan/preview")
def plan_preview(date: str, instrument: str = "LLAMAS", moon: str = None,
                 authorization: str = Header(None)):
    # preview is collaboration-wide; still require a valid key. LLAMAS and LDSS3
    # are parallel systems, so a preview is for one instrument at a time.
    return _guard(lambda: (svc.program_for(_key(authorization)),
                           svc.plan_preview(date, moon, instrument))[1])
