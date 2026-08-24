"""Tests for per-target airmass ranges (stamped #5): API validation/storage,
planner hard-constraint windows (max-only AND min-only), the standards-as-
pseudo-targets helper, and the range-aware coordinate dedupe."""
import importlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

from orchestrator.config import LLAMASConfig
from orchestrator.models import Target as OTarget
from orchestrator.planner import (_airmass_bounds, _am_ok, _find_window,
                                  compute_observability)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# API: submission, validation, dashboard passthrough
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


def _item(name, **over):
    d = {"name": name, "ra": 150.0, "dec": -29.0, "priority": "P2",
         "instrument": "LLAMAS", "mag": 13.0, "exposure_minutes": 6}
    d.update(over)
    return d


def test_submit_with_range_stored_and_returned(client):
    r = client.post("/v1/targets", headers=STUBBS, json=[
        _item("std-a", airmass_min=1.3, airmass_max=1.7)])
    assert r.status_code == 200 and r.json()[0]["status"] == "ok"
    rows = client.get("/v1/targets", headers=STUBBS).json()
    assert rows[0]["airmass_min"] == 1.3 and rows[0]["airmass_max"] == 1.7
    # rangeless target: both null (default = minimize airmass)
    client.post("/v1/targets", headers=STUBBS, json=[_item("sn", ra=10.0)])
    rows = client.get("/v1/targets", headers=STUBBS).json()
    plain = next(t for t in rows if t["name"] == "sn")
    assert plain["airmass_min"] is None and plain["airmass_max"] is None


def test_range_validation(client):
    # min >= max is nonsense
    bad = client.post("/v1/targets", headers=STUBBS, json=[
        _item("bad", airmass_min=1.7, airmass_max=1.3)])
    assert bad.json()[0]["status"] == "error"
    assert "airmass_min" in bad.json()[0]["error"]
    # airmass < 1 is unphysical
    bad2 = client.post("/v1/targets", headers=STUBBS, json=[
        _item("bad2", airmass_min=0.5)])
    assert bad2.json()[0]["status"] == "error"
    # PATCH into an invalid pair is rejected (422 via ValueError)
    ok = client.post("/v1/targets", headers=STUBBS, json=[
        _item("p", airmass_min=1.2, airmass_max=1.6)])
    tid = ok.json()[0]["id"]
    assert client.patch(f"/v1/targets/{tid}", headers=STUBBS,
                        json={"airmass_min": 2.0}).status_code == 422


def test_same_coords_distinct_bins_do_not_upsert(client):
    # one standard, two airmass bins -> TWO queue entries (dedup is per
    # airmass-range spec), and re-sending a bin upserts it
    r = client.post("/v1/targets", headers=STUBBS, json=[
        _item("GD71@am1.0-1.3", airmass_min=1.0, airmass_max=1.3),
        _item("GD71@am1.3-1.7", airmass_min=1.3, airmass_max=1.7)])
    res = r.json()
    assert [x["status"] for x in res] == ["ok", "ok"]
    assert res[0]["id"] != res[1]["id"]
    assert not res[0]["updated"] and not res[1]["updated"]
    again = client.post("/v1/targets", headers=STUBBS, json=[
        _item("GD71@am1.0-1.3", airmass_min=1.0, airmass_max=1.3)])
    assert again.json()[0]["updated"] is True
    assert again.json()[0]["id"] == res[0]["id"]


# ---------------------------------------------------------------------------
# Planner: hard-constraint windows
# ---------------------------------------------------------------------------

CFG = LLAMASConfig()
EVENING = Time("2026-08-16T01:00:00")
MORNING = Time("2026-08-16T09:00:00")


def _zenith_ra_deg():
    """RA (deg) transiting mid-window at LCO -> passes within ~deg of zenith
    for dec = site latitude."""
    mid = EVENING + (MORNING - EVENING) / 2
    return float(mid.sidereal_time("apparent",
                                   longitude=CFG.location.lon).deg)


def test_find_window_max_only_and_band():
    ra = _zenith_ra_deg()
    coord = SkyCoord(ra=ra * u.deg, dec=CFG.latitude * u.deg)  # transits at zenith
    # max-only: the classic window, best point near airmass 1
    tr, min_am, ws, we = _find_window(coord, EVENING, MORNING,
                                      CFG.location, max_airmass=1.5)
    assert tr is not None and min_am == pytest.approx(1.0, abs=0.02)
    # high-airmass band [1.4, 2.0]: the best ALLOWED point is inside the
    # band, not at the (excluded) zenith transit
    tr2, min_am2, ws2, we2 = _find_window(coord, EVENING, MORNING,
                                          CFG.location, max_airmass=2.0,
                                          min_airmass=1.4)
    assert tr2 is not None
    assert 1.4 <= min_am2 <= 2.0
    # the envelope brackets the excluded transit (non-contiguous band)
    assert ws2 < tr < we2


def test_range_overrides_global_airmass_limit():
    # a target culminating at airmass ~1.8 from LCO: alt = asin(1/1.8) ->
    # zenith distance ~56.3 deg -> dec = lat + 56.3 (northern sky)
    dec = CFG.latitude + math.degrees(math.asin(1 / 1.8) - math.pi / 2) * -1
    t_plain = OTarget(name="plain", ra_deg=_zenith_ra_deg(), dec_deg=dec,
                      exposure_minutes=10.0)
    t_range = OTarget(name="binned", ra_deg=_zenith_ra_deg(), dec_deg=dec,
                      exposure_minutes=10.0,
                      airmass_min=1.7, airmass_max=2.3)
    # global limit 1.6: the plain target never gets low enough -> dropped;
    # the explicit range OVERRIDES the global limit -> kept
    obs = compute_observability([t_plain, t_range], EVENING, MORNING, CFG)
    names = [t.name for t in obs]
    assert "plain" not in names and "binned" in names
    got = obs[0]
    assert 1.7 <= got.min_airmass <= 2.3


def test_airmass_bounds_and_slot_check():
    t_none = OTarget(name="a", ra_deg=0, dec_deg=0)
    assert _airmass_bounds(t_none, CFG) == (1.0, CFG.max_airmass)
    t_max = OTarget(name="b", ra_deg=0, dec_deg=0, airmass_max=1.5)
    assert _airmass_bounds(t_max, CFG) == (1.0, 1.5)
    t_min = OTarget(name="c", ra_deg=0, dec_deg=0, airmass_min=1.8)
    lo, hi = _airmass_bounds(t_min, CFG)
    assert lo == 1.8 and hi == float("inf")     # min-only: top unbounded
    # per-slot enforcement (catches the mid-window violation of a min band)
    assert not _am_ok(t_min, 1.2, CFG)
    assert _am_ok(t_min, 2.5, CFG)
    assert _am_ok(t_max, 1.4, CFG) and not _am_ok(t_max, 1.55, CFG)


# ---------------------------------------------------------------------------
# Standards-as-pseudo-targets helper
# ---------------------------------------------------------------------------

def _load_script():
    path = os.path.join(_REPO, "scripts", "enqueue_standards.py")
    spec = importlib.util.spec_from_file_location("enqueue_standards", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pseudo_target_naming_and_build():
    mod = _load_script()
    bins = mod.parse_bins("1.0-1.3,1.3-1.7,1.7-2.3")
    stds = mod.load_standards(os.path.join(_REPO, "ref", "standards_example.csv"))
    items = mod.build_pseudo_targets(stds, bins, priority="P2",
                                     exposure_minutes=6.0)
    assert len(items) == 2 * 3
    names = [i["name"] for i in items]
    assert "FAKE-STD1@am1.0-1.3" in names and "FAKE-STD2@am1.7-2.3" in names
    first = items[0]
    assert (first["airmass_min"], first["airmass_max"]) == (1.0, 1.3)
    assert first["priority"] == "P2" and first["instrument"] == "LLAMAS"
    # per-row exposure column wins over the CLI default
    std1 = next(i for i in items if i["name"].startswith("FAKE-STD1"))
    assert std1["exposure_minutes"] == 6.0
    # bad bins rejected
    with pytest.raises(ValueError):
        mod.parse_bins("0.8-1.3")
    with pytest.raises(ValueError):
        mod.parse_bins("1.7-1.3")


def test_loads_the_shipped_catalogue_with_its_comment_header():
    """The real Boyd et al. 2025 file carries a '#' provenance header — the
    loader must skip it rather than reading it as the field names (which made
    the documented nightly command crash with KeyError: 'name')."""
    mod = _load_script()
    stds = mod.load_standards(
        os.path.join(_REPO, "ref", "boyd2025_wdfs_standards.csv"))
    assert len(stds) == 35
    names = {s["name"] for s in stds}
    assert {"GD71", "GD153", "G191B2B"} <= names
    gd71 = next(s for s in stds if s["name"] == "GD71")
    assert gd71["ra"] == pytest.approx(88.115437, abs=1e-6)
    assert gd71["dec"] == pytest.approx(15.886239, abs=1e-6)
    assert gd71["mag"] == pytest.approx(13.0, abs=1e-3)


def test_names_filter_selects_a_nightly_subset():
    """The nightly workflow is one or two RA-appropriate standards, so --names
    must subset the catalogue (order preserved, case-insensitive) and reject an
    unknown name instead of silently dropping it."""
    mod = _load_script()
    stds = mod.load_standards(
        os.path.join(_REPO, "ref", "boyd2025_wdfs_standards.csv"))
    picked = mod.select_names(stds, "gd153,GD71")
    assert [s["name"] for s in picked] == ["GD153", "GD71"]
    assert len(mod.select_names(stds, None)) == 35
    items = mod.build_pseudo_targets(picked, mod.parse_bins("1.0-1.3,1.3-1.7"),
                                     exposure_minutes=6.0)
    assert len(items) == 2 * 2
    with pytest.raises(ValueError):
        mod.select_names(stds, "GD71,NOT-A-STANDARD")


def test_dropped_targets_are_reported_with_a_reason():
    """compute_observability must be able to say WHY it filtered a target.

    Without the out-parameter an unreachable airmass bin vanished from both the
    timeline and the overflow — submitted, then silently gone.
    """
    dec = CFG.latitude + math.degrees(math.asin(1 / 1.8) - math.pi / 2) * -1
    # asks for a low-airmass bin this target never reaches (it culminates ~1.8)
    unreachable = OTarget(name="unreachable", ra_deg=_zenith_ra_deg(),
                          dec_deg=dec, exposure_minutes=10.0,
                          airmass_min=1.0, airmass_max=1.3)
    reachable = OTarget(name="reachable", ra_deg=_zenith_ra_deg(),
                        dec_deg=dec, exposure_minutes=10.0,
                        airmass_min=1.7, airmass_max=2.3)
    dropped: list = []
    obs = compute_observability([unreachable, reachable], EVENING, MORNING,
                               CFG, dropped=dropped)
    assert [t.name for t in obs] == ["reachable"]
    assert len(dropped) == 1
    tgt, reason = dropped[0]
    assert tgt.name == "unreachable"
    assert "1.0-1.3" in reason            # names the range it could not reach
    # backward compatible: callers that pass no list still just get the survivors
    assert len(compute_observability([unreachable, reachable], EVENING,
                                     MORNING, CFG)) == 1


def test_unreachable_bin_appears_in_the_plan_overflow(client):
    """A bin the star cannot reach must show up in the plan's overflow with a
    reason — the demo failure mode was 6 submitted, 4 planned, 0 explained."""
    # GD71 (dec +15.9) from LCO in September: reaches the 1.7-2.3 band but
    # never the low ones.
    for lo, hi in ((1.0, 1.3), (1.7, 2.3)):
        client.post("/v1/targets", headers=STUBBS, json=[
            _item(f"GD71@am{lo}-{hi}", ra=88.115437, dec=15.886239,
                  exposure_minutes=10, airmass_min=lo, airmass_max=hi)])
    plan = client.get("/v1/plan/preview?date=2026-09-06&instrument=LLAMAS",
                      headers=STUBBS).json()
    scheduled = {e["target"] for e in plan["timeline"]}
    assert "GD71@am1.7-2.3" in scheduled
    over = {o["target"]: o["reason"] for o in plan["overflow"]}
    assert "GD71@am1.0-1.3" in over
    assert "1.0-1.3" in over["GD71@am1.0-1.3"]
    # nothing may be silently absent: every submission is scheduled or explained
    assert scheduled | set(over) == {"GD71@am1.0-1.3", "GD71@am1.7-2.3"}


def test_plan_sheet_stamps_when_to_shoot_each_pointing(client):
    """Three pseudo-targets of one standard are three IDENTICAL pointings on
    the sheet; the notes column has to carry the window (and say so when a row
    cannot be shot at all). The 12-column convention must not change."""
    ids = []
    for lo, hi in ((1.0, 1.3), (1.7, 2.3)):
        r = client.post("/v1/targets", headers=STUBBS, json=[
            _item(f"GD71@am{lo}-{hi}", ra=88.115437, dec=15.886239,
                  exposure_minutes=10, airmass_min=lo, airmass_max=hi)])
        ids.append(r.json()[0]["id"])
    bundle = client.post("/v1/obsplan", headers=STUBBS, json={
        "date": "2026-09-06", "instrument": "LLAMAS", "target_ids": ids}).json()
    rows = [ln.split("\t") for ln in bundle["plan_txt"].strip().split("\n")]
    assert {len(r) for r in rows} == {12}      # convention preserved
    notes = {r[0]: r[10] for r in rows}
    assert "observe" in notes["GD71@am1.7-2.3"]
    assert "UT" in notes["GD71@am1.7-2.3"]
    assert "NOT OBSERVABLE" in notes["GD71@am1.0-1.3"]
