"""Tests for observation ingestion + accounting (item F, mocked FITS source):
pointing/name/unassociated association, the 1-arcmin edge, standards airmass
disambiguation, idempotent re-ingest (no double charge), the even split across
programs, burndown folding, and the mock-generator -> adapter round trip."""
import importlib
import importlib.util
import json
import os
import sys

import pytest
from astropy.time import Time
import astropy.units as u

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UTC = "2026-08-16T05:00:00"
DATE = "2026-08-16"


def _zenith_ra(utc=UTC):
    """RA (deg) on the LCO meridian at ``utc`` — near-zenith for dec=-29."""
    from config import OBSERVATORY_CONFIG as OC
    return float(Time(utc).sidereal_time(
        "apparent", longitude=OC.longitude * u.deg).deg)


@pytest.fixture
def app_mod(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUPS_JSON", json.dumps({
        "CfA-Stubbs": {"key": "k-stubbs", "password": "ps"},
        "UA": {"key": "k-ua", "password": "pu"}}))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "queue.db"))
    monkeypatch.setenv("SEED_DEMO", "0")
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("MAGNETS_ALLOCATIONS", "ref/allocations_LLAMAS_2026B.yaml")
    monkeypatch.setenv("WARM_CACHE", "0")
    import api.app as mod
    importlib.reload(mod)
    return mod


@pytest.fixture
def client(app_mod):
    from starlette.testclient import TestClient
    return TestClient(app_mod.app)


STUBBS = {"Authorization": "Bearer k-stubbs"}
UA = {"Authorization": "Bearer k-ua"}


def _enqueue(client, name, ra, dec, headers=STUBBS, **over):
    item = {"name": name, "ra": ra, "dec": dec, "priority": "P2",
            "instrument": "LLAMAS", "mag": 19.0, "exposure_minutes": 30}
    item.update(over)
    r = client.post("/v1/targets", headers=headers, json=[item])
    assert r.status_code == 200 and r.json()[0]["status"] == "ok", r.json()
    return r.json()[0]["id"]


def _obs(filename, ra, dec, name="", exptime=600.0, utc=UTC):
    return {"utc_start": utc, "ra": ra, "dec": dec, "object_name_raw": name,
            "exptime_s": exptime, "instrument": "LLAMAS", "filename": filename}


def test_association_pointing_edge_name_unassociated(client):
    ra = _zenith_ra()
    _enqueue(client, "SN-A", ra, -29.0)
    assert client.post("/v1/observations",
                       json={"observations": []}).status_code in (401, 422)
    # unauthenticated ingest rejected
    assert client.post("/v1/observations",
                       json={"observations": [_obs("x.fits", ra, -29.0)]}
                       ).status_code == 401

    res = client.post("/v1/observations", headers=STUBBS, json={"observations": [
        _obs("a1.fits", ra, -29.0),                       # dead-on -> pointing
        _obs("a2.fits", ra, -29.0 + 59.0 / 3600.0),       # 59" -> inside 1'
        _obs("a3.fits", ra, -29.0 + 61.0 / 3600.0),       # 61" -> outside
        _obs("a4.fits", None, None, name="  sn-a  "),     # name fallback
        _obs("a5.fits", ra + 30.0, -29.0, name="whoknows"),  # nothing
    ]})
    assert res.status_code == 200
    out = {r["filename"]: r for r in res.json()["results"]}
    assert out["a1.fits"]["assoc_method"] == "pointing"
    assert out["a1.fits"]["target_name"] == "SN-A"
    assert out["a1.fits"]["airmass"] == pytest.approx(1.0, abs=0.05)
    assert out["a2.fits"]["assoc_method"] == "pointing"
    assert out["a3.fits"]["assoc_method"] == "unassociated"
    assert out["a4.fits"]["assoc_method"] == "name"
    assert out["a4.fits"]["programs"] == ["CfA-Stubbs"]
    assert out["a5.fits"]["assoc_method"] == "unassociated"
    assert res.json()["n_associated"] == 3

    # the night is queryable back
    night = client.get("/v1/observations?night=2026-08-16", headers=STUBBS).json()
    assert night["night_stamp"] == "ut20260816"
    assert len(night["observations"]) == 5


def test_standards_airmass_bin_disambiguation(client, app_mod):
    from api.observations import airmass_at
    ra = _zenith_ra()
    dec = -29.0                                 # transits zenith mid-night
    _enqueue(client, "GD71@am1.0-1.3", ra, dec,
             airmass_min=1.0, airmass_max=1.3, exposure_minutes=6)
    _enqueue(client, "GD71@am1.3-1.7", ra, dec,
             airmass_min=1.3, airmass_max=1.7, exposure_minutes=6)
    # find a real time tonight when the star sits in each bin
    def time_in(lo, hi):
        for step in range(40):
            utc = f"{DATE}T{int(1 + step * 0.25):02d}:{int((step * 0.25) % 1 * 60):02d}:00"
            am = airmass_at(ra, dec, utc)
            if am is not None and lo <= am <= hi:
                return utc, am
        raise AssertionError(f"no time in airmass bin {lo}-{hi}")
    utc_low, am_low = time_in(1.0, 1.3)
    utc_high, am_high = time_in(1.3, 1.7)
    res = client.post("/v1/observations", headers=STUBBS, json={"observations": [
        _obs("std1.fits", ra, dec, name="GD71", exptime=360, utc=utc_low),
        _obs("std2.fits", ra, dec, name="GD71", exptime=360, utc=utc_high),
    ]}).json()["results"]
    assert res[0]["target_name"] == "GD71@am1.0-1.3"
    assert res[1]["target_name"] == "GD71@am1.3-1.7"
    assert res[0]["assoc_method"] == res[1]["assoc_method"] == "pointing"


def test_idempotent_reingest_no_double_charge(client, app_mod):
    ra = _zenith_ra()
    _enqueue(client, "SN-B", ra, -29.0)
    batch = {"observations": [_obs("b1.fits", ra, -29.0, exptime=1800)]}
    assert client.post("/v1/observations", headers=STUBBS,
                       json=batch).status_code == 200
    used1 = app_mod.observation_store.used_hours_by_program()
    assert used1["CfA-Stubbs"]["LLAMAS"] == pytest.approx(0.5)
    # re-ingest the SAME filename: charges replaced, not accumulated
    assert client.post("/v1/observations", headers=STUBBS,
                       json=batch).status_code == 200
    used2 = app_mod.observation_store.used_hours_by_program()
    assert used2 == used1
    # and only one observation row exists
    rows = client.get("/v1/observations?night=2026-08-16",
                      headers=STUBBS).json()["observations"]
    assert len([r for r in rows if r["filename"] == "b1.fits"]) == 1


def test_even_split_across_two_programs(client, app_mod):
    ra = _zenith_ra()
    _enqueue(client, "SHARED-SN", ra, -20.0, headers=STUBBS)
    _enqueue(client, "SHARED-SN", ra, -20.0, headers=UA)
    res = client.post("/v1/observations", headers=STUBBS, json={"observations": [
        _obs("s1.fits", ra, -20.0, exptime=3600)]}).json()["results"][0]
    assert sorted(res["programs"]) == ["CfA-Stubbs", "UA"]
    used = app_mod.observation_store.used_hours_by_program()
    assert used["CfA-Stubbs"]["LLAMAS"] == pytest.approx(0.5)   # half each
    assert used["UA"]["LLAMAS"] == pytest.approx(0.5)


def test_burndown_folds_observed_time(client, app_mod):
    ra = _zenith_ra()
    _enqueue(client, "SN-C", ra, -29.0)
    client.post("/v1/observations", headers=STUBBS, json={"observations": [
        _obs("c1.fits", ra, -29.0, exptime=7200)]})
    from api.scheduler_bridge import load_allocations_overview
    alloc = app_mod._fold_observed_time(load_allocations_overview())
    assert alloc["tracked"] is True
    row = alloc["programs"]["CfA-Stubbs"]["LLAMAS"]
    assert row["used"] == pytest.approx(2.0)
    assert row["remaining"] == pytest.approx(row["initial"] - 2.0)
    # untouched programs keep used = 0
    assert alloc["programs"]["UA"]["LLAMAS"]["used"] == 0.0


def _load_script(fname):
    path = os.path.join(_REPO, "scripts", fname)
    spec = importlib.util.spec_from_file_location(fname[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_mock_generator_roundtrip(tmp_path):
    """mock_observations -> FITS -> ingest_fits_night adapter -> association."""
    from api.service import TargetQueueService
    from api.observations import ObservationStore
    db = str(tmp_path / "queue.db")
    svc = TargetQueueService({"k": "CfA-Stubbs"},
                             "ref/allocations_LLAMAS_2026B.yaml", db_path=db)
    ra = _zenith_ra()
    subs = [{"name": f"SN-{i}", "ra": ra + 2 * i, "dec": -29.0, "priority": "P1",
             "instrument": "LLAMAS", "mag": 19.0, "exposure_minutes": 30}
            for i in range(4)]
    subs += [{"name": "FAKE-STD@am1.0-1.3", "ra": ra, "dec": -35.0,
              "priority": "P2", "instrument": "LLAMAS", "mag": 13.0,
              "exposure_minutes": 6, "airmass_min": 1.0, "airmass_max": 1.3},
             {"name": "FAKE-STD@am1.3-1.7", "ra": ra, "dec": -35.0,
              "priority": "P2", "instrument": "LLAMAS", "mag": 13.0,
              "exposure_minutes": 6, "airmass_min": 1.3, "airmass_max": 1.7}]
    assert all(r["status"] == "ok" for r in svc.submit("k", subs))

    mock = _load_script("mock_observations.py")
    out_dir = str(tmp_path / "fits")
    files = mock.generate(db, DATE, out_dir)
    # 4 triplets + up to 2 standard bins + 1 unassociated pointing
    assert len(files) >= 4 * 3 + 1

    adapter = _load_script("ingest_fits_night.py")
    records = adapter.load_night_dir(out_dir)
    assert len(records) == len(files)
    # sexagesimal headers round-trip to the queue coordinates
    r0 = next(r for r in records if r["object_name_raw"] == "SN-0")
    assert r0["ra"] == pytest.approx(ra, abs=1e-3)
    assert r0["dec"] == pytest.approx(-29.0, abs=1e-3)

    store = ObservationStore(db_path=db)
    results = store.ingest(svc._targets, records)
    by_name: dict = {}
    for r in results:
        by_name.setdefault(r.get("target_name"), []).append(r)
    for i in range(4):                       # each science triplet associates
        assert len(by_name.get(f"SN-{i}", [])) == 3
        assert all(x["assoc_method"] == "pointing" for x in by_name[f"SN-{i}"])
    # the offset pointing stays unassociated
    unassoc = [r for r in results if r["assoc_method"] == "unassociated"]
    assert len(unassoc) == 1
    # standards (if the night geometry allowed them) land in their own bins
    std_hits = [r for r in results
                if r.get("target_name", "") and "@am" in (r.get("target_name") or "")]
    for r in std_hits:
        assert r["assoc_method"] == "pointing"
