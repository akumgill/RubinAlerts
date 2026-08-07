"""End-to-end HTTP tests of the FastAPI queue API.

Unlike test_api_service.py (which drives the service layer directly), these hit
the REAL endpoints through Starlette's TestClient — routing, bearer/cookie auth,
JSON (de)serialization, status codes. The demo flow: add targets, see them in
the queue at the right priority, reprioritize, and remove — plus auth and
per-program write-scoping. Exercises the shared 1-5 scale (P0 mandatory +
P1..P5). Requires httpx (test-only; not a production/api dependency).
"""
import importlib
import json

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUPS_JSON", json.dumps({
        "CfA-Stubbs": {"key": "k-stubbs", "password": "ps"},
        "UA": {"key": "k-ua", "password": "pu"}}))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setenv("SEED_DEMO", "0")
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("MAGNETS_ALLOCATIONS", "ref/allocations_LLAMAS_2026B.yaml")
    import api.app as app_mod
    importlib.reload(app_mod)          # rebuild svc/GROUPS from the test env
    from starlette.testclient import TestClient
    return TestClient(app_mod.app)


STUBBS = {"Authorization": "Bearer k-stubbs"}
UA = {"Authorization": "Bearer k-ua"}


def test_add_in_queue_reprioritize_remove(client):
    # unauthenticated read AND write are rejected
    assert client.get("/v1/queue").status_code == 401
    assert client.post("/v1/targets", json=[]).status_code == 401

    # ADD three targets at different rungs of the 1-5 scale
    r = client.post("/v1/targets", headers=STUBBS, json=[
        {"name": "hi", "ra": 10.0, "dec": -5.0, "priority": "P1", "instrument": "LLAMAS", "mag": 19.0},
        {"name": "mid", "ra": 11.0, "dec": -5.0, "priority": "P3", "instrument": "LLAMAS", "mag": 19.0},
        {"name": "low", "ra": 12.0, "dec": -5.0, "priority": "P5", "instrument": "LLAMAS", "mag": 19.0},
    ])
    assert r.status_code == 200
    results = r.json()
    assert all(x["status"] == "ok" for x in results)
    ids = dict(zip(["hi", "mid", "low"], [x["id"] for x in results]))

    # IN QUEUE at the right priority (shared 1-5 counts)
    counts = client.get("/v1/queue", headers=STUBBS).json()["CfA-Stubbs"]["counts"]
    assert counts["P1"] == 1 and counts["P3"] == 1 and counts["P5"] == 1

    # REPRIORITIZE the lowest up to P0 (mandatory) via PATCH
    p = client.patch(f"/v1/targets/{ids['low']}", headers=STUBBS, json={"priority": "P0"})
    assert p.status_code == 200
    counts = client.get("/v1/queue", headers=STUBBS).json()["CfA-Stubbs"]["counts"]
    assert counts["P0"] == 1 and counts["P5"] == 0

    # WRITE-SCOPING: UA cannot patch or delete a Stubbs target (404, not 200)
    assert client.patch(f"/v1/targets/{ids['hi']}", headers=UA,
                        json={"priority": "P2"}).status_code == 404
    assert client.delete(f"/v1/targets/{ids['hi']}", headers=UA).status_code == 404

    # REMOVE (withdraw) one of our own
    assert client.delete(f"/v1/targets/{ids['mid']}", headers=STUBBS).status_code == 200
    counts = client.get("/v1/queue", headers=STUBBS).json()["CfA-Stubbs"]["counts"]
    assert counts["P3"] == 0

    # a priority off the 1-5 scale is rejected per-item (batch still 200)
    bad = client.post("/v1/targets", headers=STUBBS, json=[
        {"name": "bad", "ra": 13.0, "dec": -5.0, "priority": "P9", "instrument": "LLAMAS", "mag": 19.0}])
    assert bad.status_code == 200 and bad.json()[0]["status"] == "error"


def test_multiprogram_shared_queue_deterministic(client):
    # Two accounts submit to the shared queue for a FIXED, observable night
    # (coords near LCO zenith on 2026-08-15 -> all well up, low airmass, so
    # priority drives the order deterministically). Budget is uniform/full for
    # both programs (no usage), so the outcome is fully determined by priority +
    # observability.
    assert client.post("/v1/targets", headers=STUBBS, json=[
        {"name": "S-hi", "ra": 285.0, "dec": -30.0, "priority": "P1", "instrument": "LLAMAS", "mag": 19.0},
        {"name": "S-lo", "ra": 290.0, "dec": -30.0, "priority": "P4", "instrument": "LLAMAS", "mag": 19.0},
    ]).status_code == 200
    assert client.post("/v1/targets", headers=UA, json=[
        {"name": "U-hi", "ra": 295.0, "dec": -30.0, "priority": "P1", "instrument": "LLAMAS", "mag": 19.0},
        {"name": "U-lo", "ra": 300.0, "dec": -30.0, "priority": "P3", "instrument": "LLAMAS", "mag": 19.0},
    ]).status_code == 200

    p = client.get("/v1/dashboard?date=2026-08-15&instrument=LLAMAS", headers=STUBBS).json()["plan"]
    order = [e["target"] for e in p["timeline"]]
    tiers = [e["tier"] for e in p["timeline"]]

    assert p["n_scheduled"] == 4 and len(p["overflow"]) == 0
    # cross-account priority ordering: both P1s first, then P3, then P4
    assert tiers == ["P1", "P1", "P3", "P4"]
    # both programs share the night
    assert {e["program"] for e in p["timeline"]} == {"CfA-Stubbs", "UA"}
    # within-program order respected; the lowest rung (P4) is last
    assert order.index("S-hi") < order.index("S-lo")
    assert order.index("U-hi") < order.index("U-lo")
    assert order[-1] == "S-lo"


def test_login_cookie_auth(client):
    # program list is public (populates the login screen)
    assert client.get("/v1/programs").json()["programs"] == ["CfA-Stubbs", "UA"]
    # browser login sets a session cookie that authorizes reads without a key
    assert client.post("/login", data={"program": "CfA-Stubbs", "password": "ps"}).status_code == 200
    who = client.get("/v1/whoami")
    assert who.status_code == 200 and who.json()["program"] == "CfA-Stubbs"
    # wrong password rejected
    assert client.post("/login", data={"program": "UA", "password": "nope"}).status_code == 401
