"""Integration tests wiring the per-target ledger into the planner / nightly run.

All offline: synthetic Targets, temp-file JSON ledgers, mocked airmass (as
test_planner.py does). Nothing here contacts a broker, database, or external
API.
"""

import math
from pathlib import Path

import astropy.units as u
import pandas as pd
import pytest

from orchestrator.config import LLAMAS_CONFIG
from orchestrator.models import Target
from orchestrator.planner import calculate_twilight, create_schedule
from orchestrator.target_ledger import TargetLedger


DATE = '2026-10-15'


def _full_night_target(name, ra_deg, dec_deg, evening, morning,
                       exp_min=20.0, priority=1):
    """Target near the LCO zenith with a full-night observable window."""
    t = Target(name=name, ra_deg=ra_deg, dec_deg=dec_deg, priority=priority,
               exposure_minutes=exp_min, moon_constraint='any')
    t.required_minutes_full = exp_min
    t.transit_time = evening + (morning - evening) / 2
    t.window_start = evening
    t.window_end = morning
    t.window_hours = (morning - evening).to(u.hour).value
    return t


@pytest.fixture
def _pin_airmass(monkeypatch):
    """Pin airmass low and constant so scheduling is deterministic."""
    import orchestrator.planner as planner_mod
    monkeypatch.setattr(planner_mod, '_get_airmass',
                        lambda coord, time, location: 1.1)


# ---------------------------------------------------------------------------
# create_schedule charges the ledger
# ---------------------------------------------------------------------------

def test_create_schedule_charges_ledger(tmp_path, _pin_airmass):
    """Scheduling with a ledger increments cumulative integration by the SAME
    science value (charged_minutes) billed to the program."""
    evening, morning = calculate_twilight(DATE)
    t = _full_night_target('T1', 20.0, -29.0, evening, morning, exp_min=20.0)
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))

    plan = create_schedule([t], evening, morning, moon_phase='dark',
                           prioritizer_scores={'T1': 100.0}, ledger=ledger)

    assert len(plan.scheduled) == 1
    entry = plan.scheduled[0]
    charged_min = entry.charged_minutes
    assert math.isfinite(charged_min) and charged_min > 0
    # Ledger cumulative matches the charged science time.
    assert ledger.cumulative_seconds(t) == pytest.approx(charged_min * 60.0)
    # And it persisted.
    assert Path(ledger.state_path).exists()


# ---------------------------------------------------------------------------
# Partial target schedules only remaining time
# ---------------------------------------------------------------------------

def test_partial_target_uses_remaining_time(tmp_path, _pin_airmass):
    """A target already 40/60 min done is set to its 20-min remainder and
    schedules only that much new integration."""
    evening, morning = calculate_twilight(DATE)
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))

    # Pre-load 40 min on this position.
    seed = Target(name='T1', ra_deg=20.0, dec_deg=-29.0)
    ledger.charge(seed, science_seconds=40 * 60.0, date='prev',
                  required_seconds=60 * 60.0)

    required_full = 60.0
    t = _full_night_target('T1', 20.0, -29.0, evening, morning,
                           exp_min=required_full)
    # Mimic run_nightly: set exposure_minutes to the remaining time.
    t.exposure_minutes = ledger.remaining_minutes(t, required_full)
    assert t.exposure_minutes == pytest.approx(20.0)

    plan = create_schedule([t], evening, morning, moon_phase='dark',
                           prioritizer_scores={'T1': 100.0}, ledger=ledger)

    entry = plan.scheduled[0]
    # Newly charged science ~= remaining (20 min) + overhead, not the full 60.
    assert entry.charged_minutes < 30.0
    # Cumulative now near the full required (within a block).
    assert ledger.cumulative_seconds(t) / 60.0 >= 55.0


# ---------------------------------------------------------------------------
# Multi-night accumulation -> excluded the 3rd night
# ---------------------------------------------------------------------------

def test_target_excluded_after_two_nights(tmp_path):
    """A target observed across two nightly runs accumulates past the satisfied
    threshold and is excluded (in plan.completed) on the third run."""
    from orchestrator.run_nightly import run_nightly

    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    ledger_path = str(out_dir / 'target_ledger.json')

    # Minimal candidates CSV (RubinAlerts format): one bright equatorial target.
    df = pd.DataFrame({
        'object_id': ['SN2026aa'],
        'ra': [20.0],
        'dec': [-29.0],
        'merit': [0.9],
        'mag': [19.0],   # bright -> short required exposure
    })
    csv = out_dir / 'candidates.csv'
    df.to_csv(csv, index=False)

    # Allocations YAML.
    alloc = out_dir / 'alloc.yaml'
    alloc.write_text(
        "semester: 2026B\n"
        "default_program: default\n"
        "programs:\n"
        "  - program: default\n"
        "    pi: test\n"
        "    allocated_hours: {dark: 100.0, grey: 100.0, bright: 100.0}\n"
    )

    def _run():
        return run_nightly(
            date=DATE, candidates_path=str(csv), allocations_path=str(alloc),
            moon_phase='dark', output_dir=str(out_dir),
            from_rubinalerts=True, target_ledger_path=ledger_path)

    p1 = _run()
    p2 = _run()
    p3 = _run()

    # First night schedules it; by the third it is excluded as completed.
    names1 = [e.target.name for e in p1.scheduled]
    assert 'SN2026aa' in names1

    completed3 = [t.name for t in p3.completed]
    scheduled3 = [e.target.name for e in p3.scheduled]
    assert 'SN2026aa' in completed3
    assert 'SN2026aa' not in scheduled3


# ---------------------------------------------------------------------------
# reconcile-target via the ledger API
# ---------------------------------------------------------------------------

def test_reconcile_target_adjusts_cumulative(tmp_path):
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t = Target(name='SN2026bb', ra_deg=20.0, dec_deg=-29.0)
    ledger.charge(t, science_seconds=30 * 60.0, date=DATE,
                  required_seconds=60 * 60.0)

    # Actually only integrated 21 min.
    delta = ledger.reconcile(t, actual_seconds=21 * 60.0, date=DATE)
    assert delta == pytest.approx(-9 * 60.0)
    assert ledger.cumulative_seconds(t) == pytest.approx(21 * 60.0)


# ---------------------------------------------------------------------------
# No single-night regression: first-ever run == ledger-less behavior
# ---------------------------------------------------------------------------

def test_first_run_no_ledger_regression(tmp_path, _pin_airmass):
    """A FIRST-EVER schedule (empty/no ledger file) produces the same scheduled
    targets as the ledger-less path: no single-night regression."""
    evening, morning = calculate_twilight(DATE)

    def _targets():
        return [
            _full_night_target('A', 20.0, -29.0, evening, morning,
                               exp_min=20.0, priority=1),
            _full_night_target('B', 24.0, -29.0, evening, morning,
                               exp_min=20.0, priority=2),
        ]

    scores = {'A': 120.0, 'B': 100.0}

    # Ledger-less reference.
    ref = create_schedule(_targets(), evening, morning, moon_phase='dark',
                          prioritizer_scores=scores)
    ref_names = [e.target.name for e in ref.scheduled]

    # First-ever ledger (empty).
    ledger = TargetLedger.load(str(tmp_path / 'fresh_ledger.json'))
    with_ledger = create_schedule(_targets(), evening, morning,
                                  moon_phase='dark', prioritizer_scores=scores,
                                  ledger=ledger)
    led_names = [e.target.name for e in with_ledger.scheduled]

    assert led_names == ref_names
