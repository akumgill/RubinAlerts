"""Tests for non-negotiable (PI-override) mandatory-target scheduling.

A mandatory target is RESERVED before the greedy fill: it is guaranteed a slot
when physically observable, bypasses the moon-phase filter, but cannot beat the
physics (a target that never reaches the airmass limit cannot be scheduled).

In-memory: Targets built directly with windows set the way test_planner.py does;
airmass is monkeypatched constant so ephemeris swings don't interfere. No
network.
"""

import logging

import astropy.units as u

from orchestrator.models import Target
from orchestrator.planner import calculate_twilight, create_schedule


DATE = '2026-10-15'


def _windowed_target(name, ra_deg, dec_deg, evening, morning,
                     exp_min=20.0, priority=1, moon='any', mandatory=False):
    """Build a Target with a full-night observable window populated."""
    t = Target(name=name, ra_deg=ra_deg, dec_deg=dec_deg, priority=priority,
               exposure_minutes=exp_min, moon_constraint=moon,
               mandatory=mandatory)
    t.transit_time = evening + (morning - evening) / 2
    t.window_start = evening
    t.window_end = morning
    t.window_hours = (morning - evening).to(u.hour).value
    return t


def test_low_score_mandatory_target_is_reserved(monkeypatch):
    """A mandatory target with a LOW composite score is still scheduled when
    observable, even though a higher-scoring non-mandatory target competes for
    overlapping time."""
    import orchestrator.planner as planner_mod
    monkeypatch.setattr(planner_mod, '_get_airmass',
                        lambda coord, time, location: 1.1)

    evening, morning = calculate_twilight(DATE)

    # Mandatory target with a deliberately tiny score.
    must = _windowed_target('MUST', 20.0, -29.0, evening, morning,
                            exp_min=30.0, priority=4, mandatory=True)
    # A high-scoring competitor occupying the same window.
    rich = _windowed_target('RICH', 24.0, -29.0, evening, morning,
                            exp_min=30.0, priority=1)

    scores = {'MUST': 1.0, 'RICH': 10000.0}

    plan = create_schedule(
        [rich, must], evening, morning,
        moon_phase='dark', prioritizer_scores=scores,
    )

    names = [e.target.name for e in plan.scheduled]
    assert 'MUST' in names  # reserved despite the low score
    assert plan.unschedulable_mandatory == []


def test_unobservable_mandatory_target_recorded(monkeypatch, caplog):
    """A mandatory target that never reaches the airmass limit (no observable
    window) is NOT scheduled, lands in unschedulable_mandatory, and warns."""
    import orchestrator.planner as planner_mod
    monkeypatch.setattr(planner_mod, '_get_airmass',
                        lambda coord, time, location: 1.1)

    evening, morning = calculate_twilight(DATE)

    # No window populated -> simulates a target that never reaches max airmass.
    never = Target(name='NEVER', ra_deg=10.0, dec_deg=80.0, priority=1,
                   exposure_minutes=30.0, mandatory=True)
    assert never.window_start is None

    filler = _windowed_target('FILLER', 20.0, -29.0, evening, morning,
                              exp_min=30.0, priority=2)

    with caplog.at_level(logging.WARNING):
        plan = create_schedule(
            [never, filler], evening, morning, moon_phase='dark')

    names = [e.target.name for e in plan.scheduled]
    assert 'NEVER' not in names
    assert any(t.name == 'NEVER' for t in plan.unschedulable_mandatory)
    assert any('NEVER' in r.message for r in caplog.records)


def test_mandatory_bypasses_moon_filter(monkeypatch):
    """A mandatory dark-only target on a BRIGHT night bypasses the moon filter
    and schedules (when observable)."""
    import orchestrator.planner as planner_mod
    monkeypatch.setattr(planner_mod, '_get_airmass',
                        lambda coord, time, location: 1.1)

    evening, morning = calculate_twilight(DATE)

    # dark-only constraint that would normally be filtered out on a bright night
    dark_must = _windowed_target('DARKMUST', 20.0, -29.0, evening, morning,
                                 exp_min=30.0, priority=3, moon='dark',
                                 mandatory=True)
    # A non-mandatory dark-only target, by contrast, should be filtered out.
    dark_skip = _windowed_target('DARKSKIP', 24.0, -29.0, evening, morning,
                                 exp_min=30.0, priority=1, moon='dark')

    plan = create_schedule(
        [dark_must, dark_skip], evening, morning, moon_phase='bright')

    names = [e.target.name for e in plan.scheduled]
    assert 'DARKMUST' in names           # override bypassed the moon filter
    assert 'DARKSKIP' not in names       # ordinary dark-only target excluded
    assert plan.unschedulable_mandatory == []
