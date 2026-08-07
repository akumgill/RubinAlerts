"""Tests for the target-submission API service core (no web framework).

The scheduler-bridge preview is exercised by one small smoke test; the rest
are fast in-memory checks of submit/upsert/dedup/withdraw/queue/auth.
"""
import pytest

from api import TargetQueueService
from api.service import AuthError, NotFound

KEYS = {"kA": "prog-A", "kB": "prog-B"}

RESOLVER_DB = {
    "SN Test1": {"ra": 150.0, "dec": -10.0, "mag": 18.5, "redshift": 0.05, "scheme": "TNS"},
}


def resolver(name):
    return RESOLVER_DB.get(name)


@pytest.fixture
def alloc(tmp_path):
    p = tmp_path / "alloc.yaml"
    p.write_text(
        'semester: "2026B"\n'
        'default_program: "prog-A"\n'
        'programs:\n'
        '  - {program: "prog-A", pi: "A", allocated_hours: {dark: 5, grey: 5, bright: 5}}\n'
        '  - {program: "prog-B", pi: "B", allocated_hours: {dark: 5, grey: 5, bright: 5}}\n'
    )
    return str(p)


@pytest.fixture
def svc(alloc):
    # require_exposure=False keeps these low-level submit/dedup/auth checks
    # focused on their own concern; the exposure requirement is covered by its
    # own tests below and end-to-end in test_api_http.
    return TargetQueueService(KEYS, alloc, resolver=resolver,
                              require_exposure=False)


@pytest.fixture
def svc_req(alloc):
    """Service with the production default (exposure required)."""
    return TargetQueueService(KEYS, alloc, resolver=resolver)


def test_exposure_required_by_default(svc_req):
    # missing exposure is rejected
    r = svc_req.submit("kA", [{"ra": 10, "dec": 20, "mag": 19, "priority": "P1"}])
    assert r[0]["status"] == "error" and "exposure" in r[0]["error"]
    # explicit total is accepted
    r = svc_req.submit("kA", [{"ra": 10, "dec": 20, "mag": 19, "priority": "P1",
                               "exposure_minutes": 30}])
    assert r[0]["status"] == "ok"


def test_exposure_from_subexposure_spec(svc_req):
    # n_exposures x exposure_seconds sets the total (3 x 600 s = 30 min)
    r = svc_req.submit("kA", [{"ra": 40, "dec": 20, "mag": 19, "priority": "P1",
                               "n_exposures": 3, "exposure_seconds": 600}])
    assert r[0]["status"] == "ok"
    t = svc_req.list_targets("kA")[0]
    assert t["exposure_minutes"] == 30.0
    assert t["n_exposures"] == 3 and t["exposure_seconds"] == 600.0


# ---- auth ----
def test_unknown_key_rejected(svc):
    with pytest.raises(AuthError):
        svc.submit("nope", [{"ra": 10, "dec": 10, "mag": 19, "priority": "P1"}])


def test_program_from_key(svc):
    assert svc.program_for("kA") == "prog-A"


# ---- submit / validation ----
def test_submit_with_coords(svc):
    r = svc.submit("kA", [{"ra": 10.0, "dec": 20.0, "mag": 19.0, "priority": "P1"}])
    assert r[0]["status"] == "ok" and r[0]["updated"] is False
    assert r[0]["resolved_from"] == "coords"


def test_bad_priority_rejected(svc):
    r = svc.submit("kA", [{"ra": 10, "dec": 20, "mag": 19, "priority": "P9"}])
    assert r[0]["status"] == "error"


def test_coords_only_needs_mag(svc):
    r = svc.submit("kA", [{"ra": 10, "dec": 20, "priority": "P1"}])
    assert r[0]["status"] == "error" and "mag" in r[0]["error"]


def test_batch_partial_failure(svc):
    r = svc.submit("kA", [
        {"ra": 10, "dec": 20, "mag": 19, "priority": "P1"},   # ok
        {"priority": "P1"},                                    # no coords, no name
    ])
    assert r[0]["status"] == "ok"
    assert r[1]["status"] == "error"


# ---- name resolution ----
def test_name_resolves(svc):
    r = svc.submit("kA", [{"name": "SN Test1", "priority": "P2"}])
    assert r[0]["status"] == "ok"
    assert r[0]["resolved_from"] == "name:TNS"


def test_unresolvable_name(svc):
    r = svc.submit("kA", [{"name": "SN Nonexistent", "priority": "P2"}])
    assert r[0]["status"] == "error"


# ---- upsert / dedup ----
def test_resubmit_same_coords_upserts(svc):
    r1 = svc.submit("kA", [{"ra": 10.0, "dec": 20.0, "mag": 19.0, "priority": "P3"}])
    r2 = svc.submit("kA", [{"ra": 10.00003, "dec": 20.0, "mag": 18.5, "priority": "P1"}])  # ~0.1"
    assert r2[0]["updated"] is True
    assert r1[0]["id"] == r2[0]["id"]
    t = [x for x in svc.list_targets("kA") if x["id"] == r1[0]["id"]][0]
    assert t["priority"] == "P1" and t["mag"] == 18.5
    assert len(svc.active()) == 1


def test_far_apart_not_merged(svc):
    svc.submit("kA", [{"ra": 10.0, "dec": 20.0, "mag": 19, "priority": "P1"}])
    svc.submit("kA", [{"ra": 10.1, "dec": 20.0, "mag": 19, "priority": "P1"}])  # 6' apart
    assert len(svc.active()) == 2


def test_shared_object_across_programs(svc):
    svc.submit("kA", [{"ra": 10.0, "dec": 20.0, "mag": 19, "priority": "P1"}])
    rb = svc.submit("kB", [{"ra": 10.0, "dec": 20.0, "mag": 19, "priority": "P2"}])
    assert rb[0]["shared_with"] == ["prog-A"]
    assert len(svc.active()) == 2  # each program keeps its own submission


# ---- list scoping / patch / withdraw ----
def test_list_is_program_scoped(svc):
    svc.submit("kA", [{"ra": 10, "dec": 20, "mag": 19, "priority": "P1"}])
    svc.submit("kB", [{"ra": 30, "dec": 40, "mag": 19, "priority": "P1"}])
    assert len(svc.list_targets("kA")) == 1
    assert len(svc.list_targets("kB")) == 1


def test_cannot_patch_other_program(svc):
    r = svc.submit("kA", [{"ra": 10, "dec": 20, "mag": 19, "priority": "P1"}])
    with pytest.raises(NotFound):
        svc.patch("kB", r[0]["id"], {"priority": "P2"})


def test_withdraw_removes_from_active(svc):
    r = svc.submit("kA", [{"ra": 10, "dec": 20, "mag": 19, "priority": "P1"}])
    svc.withdraw("kA", r[0]["id"])
    assert len(svc.active()) == 0


# ---- queue summary ----
def test_queue_summary_counts(svc):
    svc.submit("kA", [{"ra": 10, "dec": 20, "mag": 19, "priority": "P1"},
                      {"ra": 11, "dec": 20, "mag": 19, "priority": "P3"}])
    q = svc.queue_summary()
    assert q["prog-A"]["counts"]["P1"] == 1
    assert q["prog-A"]["counts"]["P3"] == 1
    assert q["prog-A"]["requested_hours"] > 0


# ---- instrument field ----
def test_instrument_defaults_llamas(svc):
    r = svc.submit("kA", [{"ra": 10, "dec": 20, "mag": 19, "priority": "P1"}])
    t = svc.list_targets("kA")[0]
    assert t["instrument"] == "LLAMAS"


def test_instrument_validated(svc):
    r = svc.submit("kA", [{"ra": 10, "dec": 20, "mag": 19, "priority": "P1",
                           "instrument": "GMOS"}])
    assert r[0]["status"] == "error" and "instrument" in r[0]["error"]


def test_active_filters_by_instrument(svc):
    svc.submit("kA", [
        {"name": "L1", "ra": 10, "dec": 20, "mag": 19, "priority": "P1", "instrument": "LLAMAS"},
        {"name": "D1", "ra": 30, "dec": 20, "mag": 19, "priority": "P1", "instrument": "LDSS3"},
        {"name": "E1", "ra": 50, "dec": 20, "mag": 19, "priority": "P1", "instrument": "EITHER"},
    ])
    llamas = {t.name for t in svc.active(instrument="LLAMAS")}
    ldss3 = {t.name for t in svc.active(instrument="LDSS3")}
    assert llamas == {"L1", "E1"}   # EITHER joins both universes
    assert ldss3 == {"D1", "E1"}


# ---- preview smoke test (runs the real orchestrator) ----
def test_plan_preview_smoke(svc):
    svc.submit("kA", [
        {"name": "T-A1", "ra": 210.0, "dec": -5.0, "mag": 18.5, "redshift": 0.05, "priority": "P1"},
        {"name": "T-A2", "ra": 330.0, "dec": -8.0, "mag": 19.0, "redshift": 0.07, "priority": "P2"},
    ])
    plan = svc.plan_preview("2026-07-15")
    assert plan["moon"] in ("dark", "grey", "bright")
    assert isinstance(plan["timeline"], list)
    assert "requested_hours" in plan
    # every timeline entry is tagged with a program
    assert all(e["program"] for e in plan["timeline"])


# ---- the Ia alert feed on a LLAMAS night, and instrument isolation ----
def test_ia_feed_schedules_on_llamas_night(svc):
    # a handful of Ia-like targets submitted as the automated feed (LLAMAS),
    # plus one LDSS3 target that must NOT appear on a LLAMAS-night plan.
    svc.submit("kA", [
        {"name": "SNIa-1", "ra": 340.0, "dec": 5.0, "mag": 18.4, "redshift": 0.06, "priority": "P1", "instrument": "LLAMAS"},
        {"name": "SNIa-2", "ra": 320.0, "dec": -4.0, "mag": 18.9, "redshift": 0.11, "priority": "P2", "instrument": "LLAMAS"},
    ])
    svc.submit("kB", [
        {"name": "LDSS-only", "ra": 335.0, "dec": -10.0, "mag": 19.5, "priority": "P1", "instrument": "LDSS3"},
    ])
    plan = svc.plan_preview("2026-07-15", instrument="LLAMAS")
    assert plan["instrument"] == "LLAMAS"
    names = {e["target"] for e in plan["timeline"]}
    assert names, "Ia feed produced no LLAMAS-night plan"
    assert "LDSS-only" not in names   # the LDSS3 target is in the other universe
