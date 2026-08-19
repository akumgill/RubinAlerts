"""Tests for the operator observing-plan bundle (item G): the TCS catalog
serializer against the REAL ref/march_obs_run/catalog.cat row format, the
plan-sheet golden line, the provisional LLAMAS macro, and the /v1/obsplan
endpoint (auth, 1-6 batch validation, triplet arithmetic, ordering)."""
import importlib
import json
import os

import pytest

from orchestrator.obsfiles import (dec_dms, exp_str, llamas_macro, plan_sheet,
                                   ra_hms, tcs_catalog)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# the real LTT2415 row from ref/march_obs_run/catalog.cat (line 5)
LTT2415_RA = (5 + 56 / 60 + 24.30 / 3600) * 15.0
LTT2415_DEC = -(27 + 51 / 60 + 28.80 / 3600)


def test_tcs_catalog_matches_real_row_column_for_column():
    with open(os.path.join(_REPO, "ref", "march_obs_run", "catalog.cat")) as f:
        real = f.read().splitlines()[4].split("\t")     # the LTT2415 row
    assert real[1] == "LTT2415"
    ours = tcs_catalog([{"name": "LTT2415", "ra": LTT2415_RA,
                         "dec": LTT2415_DEC, "n_exp": 3, "exp_sec": 60}])
    fields = ours.strip().split("\t")
    assert len(fields) == len(real) == 15
    # column-for-column: everything but the running index must agree
    assert fields[1:] == real[1:]
    assert fields[0] == "1"                             # ours renumbers from 1


def test_tcs_catalog_numbering_and_shape():
    rows = [{"name": f"T{i}", "ra": 10.0 * i + 1, "dec": -5.0 * i,
             "n_exp": 3, "exp_sec": 600} for i in range(3)]
    lines = tcs_catalog(rows).strip().split("\n")
    assert [l.split("\t")[0] for l in lines] == ["1", "2", "3"]
    for l in lines:
        f = l.split("\t")
        assert len(f) == 15
        assert f[4:9] == ["2000.0", "0.0", "0.0", "-62.5", "HRZ"]
        assert f[9:12] == f[12:15] == ["00:00:00.0", "+00:00:00", "2000.0"]


def test_plan_sheet_golden_line():
    row = {"name": "2026anl", "ra": 106.353417, "dec": 19.069708,
           "mag": 19.5, "priority": "P1", "instrument": "LLAMAS",
           "n_exp": 3, "exp_sec": 900, "notes": "Priority target"}
    line = plan_sheet([row], date="2026-01-23").strip()
    assert line == ("2026anl\t07:05:24.82\t+19:04:10.95\t106.353417\t"
                    "19.069708\t1\t2026-01-23\tLLAMAS\t19.5\t3x900s\t"
                    "Priority target\tN/A")


def test_llamas_macro_provisional_marker():
    row = {"name": "SN-X", "ra": 150.0, "dec": -29.0, "n_exp": 3,
           "exp_sec": 600}
    macro = llamas_macro([row])
    assert "FORMAT PROVISIONAL" in macro and "Simcoe" in macro
    assert "3 x 600s" in macro and "SN-X" in macro
    assert ra_hms(150.0) in macro and dec_dms(-29.0) in macro


def test_exp_str_triplets():
    assert exp_str({"n_exp": 3, "exp_sec": 900}) == "3x900s"
    assert exp_str({"n_exp": 1, "exp_sec": 120}) == "1x120s"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUPS_JSON", json.dumps({
        "CfA-Stubbs": {"key": "k-stubbs", "password": "ps"}}))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setenv("SEED_DEMO", "0")
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("MAGNETS_ALLOCATIONS", "ref/allocations_LLAMAS_2026B.yaml")
    monkeypatch.setenv("WARM_CACHE", "0")
    import api.app as app_mod
    importlib.reload(app_mod)
    from starlette.testclient import TestClient
    return TestClient(app_mod.app)


STUBBS = {"Authorization": "Bearer k-stubbs"}


def test_obsplan_endpoint(client):
    # auth required
    assert client.post("/v1/obsplan",
                       json={"target_ids": [1]}).status_code == 401

    ids = []
    for i, extra in enumerate([
            {"n_exposures": 3, "exposure_seconds": 1170},
            {"exposure_minutes": 30},          # -> 3 x 600 s triplet
            {"exposure_minutes": 45}]):
        item = {"name": f"SN-{i}", "ra": 280.0 + 5 * i, "dec": -29.0,
                "priority": "P1", "instrument": "LLAMAS", "mag": 19.0}
        item.update(extra)
        r = client.post("/v1/targets", headers=STUBBS, json=[item])
        assert r.json()[0]["status"] == "ok"
        ids.append(r.json()[0]["id"])

    # batch-size validation: empty and >6 rejected with the protocol message
    assert client.post("/v1/obsplan", headers=STUBBS,
                       json={"target_ids": []}).status_code == 422
    seven = client.post("/v1/obsplan", headers=STUBBS,
                        json={"target_ids": list(range(1, 8))})
    assert seven.status_code == 422 and "batches of 4-6" in seven.json()["detail"]
    # unknown id
    assert client.post("/v1/obsplan", headers=STUBBS,
                       json={"target_ids": [999]}).status_code == 422

    r = client.post("/v1/obsplan", headers=STUBBS, json={
        "target_ids": ids, "instrument": "LLAMAS", "date": "2026-08-16"})
    assert r.status_code == 200
    b = r.json()
    assert sorted(b["order"]) == ["SN-0", "SN-1", "SN-2"]
    # TCS catalog: 3 rows x 15 tab fields
    cat_lines = b["catalog_cat"].strip().split("\n")
    assert len(cat_lines) == 3
    assert all(len(l.split("\t")) == 15 for l in cat_lines)
    # triplet arithmetic: explicit spec kept; totals split into 3 x N (10 s
    # rounding)
    assert "3x1170s" in b["plan_txt"]
    assert "3x600s" in b["plan_txt"]       # 30 min -> 600 s subs
    assert "3x900s" in b["plan_txt"]       # 45 min -> 900 s subs
    assert "FORMAT PROVISIONAL" in b["instrument_macro"]
    # ordering follows tonight's observability windows: RA 280 rises (and
    # sets) before RA 290 on any night, so SN-0 precedes SN-2
    assert b["order"].index("SN-0") < b["order"].index("SN-2")
