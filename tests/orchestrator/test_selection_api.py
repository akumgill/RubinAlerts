"""HTTP tests of the SN Ia selection endpoints (POST /v1/selection/nights,
GET /v1/selection) through Starlette's TestClient — auth, the SELECTION_PROGRAMS
read allowlist, the ingest->read roundtrip, validation limits, and the
cross-night persistence computation. Requires httpx (test-only)."""
import importlib
import json

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUPS_JSON", json.dumps({
        "CfA-Stubbs": {"key": "k-stubbs", "password": "ps"},
        "CfA-Villar": {"key": "k-villar", "password": "pv"}}))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setenv("SEED_DEMO", "0")
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("MAGNETS_ALLOCATIONS", "ref/allocations_LLAMAS_2026B.yaml")
    monkeypatch.setenv("WARM_CACHE", "0")   # no background warm thread in tests
    monkeypatch.delenv("SELECTION_PROGRAMS", raising=False)  # default: CfA-Stubbs
    import api.app as app_mod
    importlib.reload(app_mod)          # rebuild svc/GROUPS from the test env
    from starlette.testclient import TestClient
    return TestClient(app_mod.app)


STUBBS = {"Authorization": "Bearer k-stubbs"}
VILLAR = {"Authorization": "Bearer k-villar"}


def _night(stamp, mjd, candidates):
    return {"night_stamp": stamp, "mjd": mjd,
            "summary": {"n_candidates": len(candidates)},
            "candidates": candidates}


def _cand(i, **over):
    c = {"diaObjectId": f"dia-{i}", "ztf_oid": f"ZTF26aaaaa{i:02d}",
         "tns_name": None, "tns_type": None, "ra": 10.0 + i, "dec": -5.0,
         "peak_mag": 19.0, "delta_t": 2.0, "z": 0.05, "merit": 1.0 - 0.1 * i,
         "n_points": 12, "surveys": "ZTF", "offset_class": "offset"}
    c.update(over)
    return c


def test_ingest_requires_auth(client):
    r = client.post("/v1/selection/nights",
                    json=_night("ut20260818", 61270.0, [_cand(0)]))
    assert r.status_code == 401
    # read is also 401 without any identity
    assert client.get("/v1/selection").status_code == 401


def test_read_allowlist(client):
    # seed one night (any valid program may upload — here the pipeline key)
    up = client.post("/v1/selection/nights", headers=STUBBS,
                     json=_night("ut20260818", 61270.0, [_cand(0)]))
    assert up.status_code == 200

    # non-allowlisted program: 403 with a clear message
    r = client.get("/v1/selection", headers=VILLAR)
    assert r.status_code == 403
    assert "CfA-Stubbs" in r.json()["detail"]
    assert "CfA-Villar" in r.json()["detail"]

    # allowlisted program: 200
    r = client.get("/v1/selection", headers=STUBBS)
    assert r.status_code == 200
    assert [n["night_stamp"] for n in r.json()["nights"]] == ["ut20260818"]


def test_roundtrip_ingest_read(client):
    # candidate 0 carries a compact light curve ([mjd, mag, magerr, band] rows)
    lc = [[61265.1234, 19.512, 0.051, "g"], [61266.4321, 19.301, None, "r"],
          [61268.9876, 19.105, 0.032, "r"]]
    cands = [_cand(0, tns_name="SN 2026xbx", tns_type="SN Ia", lc=lc), _cand(1)]
    up = client.post("/v1/selection/nights", headers=STUBBS,
                     json=_night("ut20260818", 61270.0, cands))
    assert up.status_code == 200
    assert up.json() == {"ok": True, "night_stamp": "ut20260818",
                         "n_candidates": 2}

    body = client.get("/v1/selection", headers=STUBBS).json()
    night = body["nights"][0]
    assert night["night_stamp"] == "ut20260818"
    assert night["mjd"] == 61270.0
    assert night["summary"]["n_candidates"] == 2
    assert night["candidates"][0]["tns_name"] == "SN 2026xbx"
    assert night["candidates"][0]["lc"] == lc      # photometry rides through
    assert "lc" not in night["candidates"][1]      # omitted stays omitted
    assert night["candidates"][1]["ztf_oid"] == "ZTF26aaaaa01"

    # re-upload the same stamp upserts (no duplicate night)
    up2 = client.post("/v1/selection/nights", headers=STUBBS,
                      json=_night("ut20260818", 61270.0, cands[:1]))
    assert up2.status_code == 200 and up2.json()["n_candidates"] == 1
    body = client.get("/v1/selection", headers=STUBBS).json()
    assert len(body["nights"]) == 1
    assert len(body["nights"][0]["candidates"]) == 1


def test_ingest_validation(client):
    # bad night stamp -> 422
    r = client.post("/v1/selection/nights", headers=STUBBS,
                    json=_night("2026-08-18", 61270.0, []))
    assert r.status_code == 422
    # too many candidates -> 422
    r = client.post("/v1/selection/nights", headers=STUBBS,
                    json=_night("ut20260818", 61270.0,
                                [{"diaObjectId": str(i)} for i in range(201)]))
    assert r.status_code == 422
    # payload cap is 2.5 MB (raised from 1 MB when lc photometry joined the
    # payload): just under passes, just over is 413
    under = [{"diaObjectId": str(i), "notes": "x" * 15000} for i in range(150)]
    r = client.post("/v1/selection/nights", headers=STUBBS,
                    json=_night("ut20260818", 61270.0, under))  # ~2.3 MB
    assert r.status_code == 200
    fat = [{"diaObjectId": str(i), "notes": "x" * 18000} for i in range(150)]
    r = client.post("/v1/selection/nights", headers=STUBBS,
                    json=_night("ut20260818", 61270.0, fat))    # ~2.7 MB
    assert r.status_code == 413


def test_persistence_across_nights(client):
    # night 1 (older): A ranked 3rd, B ranked 1st, C only tonight
    n1 = [_cand(0, ztf_oid="ZTF26B", merit=0.9),
          _cand(1, ztf_oid="ZTF26C", merit=0.8),
          _cand(2, ztf_oid="ZTF26A", merit=0.7)]
    # night 2 (newer): A rises to rank 1, B drops to 2; C gone; D new
    n2 = [_cand(0, ztf_oid="ZTF26A", merit=0.95, tns_name="SN 2026aaa"),
          _cand(1, ztf_oid="ZTF26B", merit=0.85),
          _cand(2, ztf_oid="ZTF26D", merit=0.5)]
    assert client.post("/v1/selection/nights", headers=STUBBS,
                       json=_night("ut20260817", 61269.0, n1)).status_code == 200
    assert client.post("/v1/selection/nights", headers=STUBBS,
                       json=_night("ut20260818", 61270.0, n2)).status_code == 200

    body = client.get("/v1/selection", headers=STUBBS).json()
    # nights come back newest first
    assert [n["night_stamp"] for n in body["nights"]] == ["ut20260818", "ut20260817"]

    pers = body["persistence"]
    # only A and B appear in >= 2 nights (C and D are single-night)
    assert [p["id"] for p in pers] == ["ZTF26A", "ZTF26B"]  # by latest rank
    a, b = pers
    assert a["latest_rank"] == 1 and a["latest_merit"] == 0.95
    assert a["tns_name"] == "SN 2026aaa"
    # appearances oldest -> newest with per-night rank + merit
    assert [(x["night_stamp"], x["rank"]) for x in a["appearances"]] == \
        [("ut20260817", 3), ("ut20260818", 1)]
    assert [(x["night_stamp"], x["rank"]) for x in b["appearances"]] == \
        [("ut20260817", 1), ("ut20260818", 2)]
    assert b["appearances"][0]["merit"] == 0.9


def test_selection_programs_env_override(client, monkeypatch, tmp_path):
    # a custom allowlist admits Villar and excludes Stubbs
    monkeypatch.setenv("SELECTION_PROGRAMS", "CfA-Villar")
    import api.app as app_mod
    importlib.reload(app_mod)
    from starlette.testclient import TestClient
    c = TestClient(app_mod.app)
    assert c.get("/v1/selection", headers=VILLAR).status_code == 200
    assert c.get("/v1/selection", headers=STUBBS).status_code == 403
