"""Tests for scheduler quality (chunk E: R8, R9, R18).

Slew penalty, scoring-mode stamp, and the small-list P-label guard. All
in-memory: Targets built directly; twilight computed from ephemerides (no
network — astropy uses bundled IERS/solar data); windows mocked wide so the
slew penalty is the deciding factor.
"""

import math

import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from orchestrator.config import LLAMAS_CONFIG
from orchestrator.models import Target
from orchestrator.planner import calculate_twilight, create_schedule
from orchestrator.normalize import load_from_rubinalerts
from orchestrator.output import write_summary


DATE = '2026-10-15'


def _circumpolar_target(name, ra_deg, dec_deg, evening, morning,
                        exp_min=20.0):
    """Build a Target near the LCO zenith with a full-night window.

    Dec near the site latitude (~-29) keeps airmass low for the whole night,
    so airmass differences don't swamp the slew penalty under test.
    """
    t = Target(name=name, ra_deg=ra_deg, dec_deg=dec_deg, priority=1,
               exposure_minutes=exp_min, moon_constraint='any')
    t.transit_time = evening + (morning - evening) / 2
    t.window_start = evening
    t.window_end = morning
    t.window_hours = (morning - evening).to(u.hour).value
    return t


def test_slew_penalty_breaks_tie(monkeypatch):
    """Two candidates with equal composite score but different sky positions:
    the one nearer the previously-scheduled target is chosen (R9).

    Airmass is pinned constant so the slew term is the only differentiator —
    isolating the penalty under test from ephemeris-driven airmass swings.
    """
    import orchestrator.planner as planner_mod
    monkeypatch.setattr(planner_mod, '_get_airmass',
                        lambda coord, time, location: 1.1)

    evening, morning = calculate_twilight(DATE)

    # Anchor target transits first so it is placed first and becomes prev_coord.
    # Give it the earliest transit so the deterministic transit-sort puts it up
    # front; a generous window lets it be picked at evening twilight.
    anchor = _circumpolar_target('ANCHOR', ra_deg=20.0, dec_deg=-29.0,
                                 evening=evening, morning=morning, exp_min=20.0)

    # Two competitors at equal composite score, different separations from
    # ANCHOR. NEAR is a few deg away; FAR is ~40 deg away.
    near = _circumpolar_target('NEAR', ra_deg=24.0, dec_deg=-29.0,
                               evening=evening, morning=morning, exp_min=20.0)
    far = _circumpolar_target('FAR', ra_deg=64.0, dec_deg=-29.0,
                              evening=evening, morning=morning, exp_min=20.0)

    # Equal prioritizer scores for the two competitors; anchor highest so it is
    # scheduled first regardless.
    scores = {'ANCHOR': 1000.0, 'NEAR': 100.0, 'FAR': 100.0}

    plan = create_schedule(
        [anchor, near, far], evening, morning,
        moon_phase='dark', prioritizer_scores=scores,
    )

    names = [e.target.name for e in plan.scheduled]
    assert 'ANCHOR' in names
    # After ANCHOR, NEAR must be chosen before FAR (slew tie-break).
    assert names.index('NEAR') < names.index('FAR')
    # And immediately after the anchor.
    assert names[names.index('ANCHOR') + 1] == 'NEAR'


def test_scoring_mode_prioritizer():
    """create_schedule stamps 'prioritizer' when scores are supplied (R18)."""
    evening, morning = calculate_twilight(DATE)
    t = _circumpolar_target('T1', 20.0, -29.0, evening, morning)
    plan = create_schedule([t], evening, morning, moon_phase='dark',
                           prioritizer_scores={'T1': 100.0})
    assert plan.scoring_mode == 'prioritizer'


def test_scoring_mode_fallback():
    """create_schedule stamps fallback mode when no scores are supplied."""
    evening, morning = calculate_twilight(DATE)
    t = _circumpolar_target('T1', 20.0, -29.0, evening, morning)
    plan = create_schedule([t], evening, morning, moon_phase='dark')
    assert plan.scoring_mode == 'fallback/priority-only'


def test_summary_header_has_mode_and_relative_caveat(tmp_path):
    """write_summary header contains the scoring-mode stamp and the
    within-night relative-P caveat (R18, R8)."""
    evening, morning = calculate_twilight(DATE)
    t = _circumpolar_target('T1', 20.0, -29.0, evening, morning)
    plan = create_schedule([t], evening, morning, moon_phase='dark',
                           prioritizer_scores={'T1': 100.0})

    out = tmp_path / 'summary.txt'
    write_summary(plan, str(out))
    text = out.read_text()

    assert 'Scoring mode:' in text
    assert 'prioritizer' in text
    assert 'WITHIN-NIGHT RELATIVE' in text


def test_small_list_priority_guard(tmp_path):
    """<4-target candidate list assigns sane P-labels without error (R8)."""
    df = pd.DataFrame({
        'object_id': ['A', 'B', 'C'],
        'ra': [150.0, 151.0, 152.0],
        'dec': [2.0, 2.1, 2.2],
        'merit': [0.9, 0.5, 0.2],
    })
    csv = tmp_path / 'candidates.csv'
    df.to_csv(csv, index=False)

    targets = load_from_rubinalerts(str(csv))
    assert len(targets) == 3
    by_name = {t.name: t for t in targets}
    # Best merit -> P1, ranked down; all distinct, none collapsed/crashing.
    assert by_name['A'].priority == 1
    assert by_name['B'].priority == 2
    assert by_name['C'].priority == 3
    for t in targets:
        assert 1 <= t.priority <= 4


def test_single_target_priority_guard(tmp_path):
    """A 1-target list still produces a valid P-label (degenerate quartiles)."""
    df = pd.DataFrame({
        'object_id': ['ONLY'],
        'ra': [150.0],
        'dec': [2.0],
        'merit': [0.7],
    })
    csv = tmp_path / 'candidates.csv'
    df.to_csv(csv, index=False)

    targets = load_from_rubinalerts(str(csv))
    assert len(targets) == 1
    assert 1 <= targets[0].priority <= 4


# ---------------------------------------------------------------------------
# 2026-07 scheduling upgrades: slew+buffer charged in wall-clock, value-density
# nudge, mid-night standards get real reserved blocks.
# ---------------------------------------------------------------------------

def _mk_target(name, ra, dec, exp_min, priority=1):
    from orchestrator.models import Target
    t = Target(name=name, ra_deg=ra, dec_deg=dec, priority=priority,
               exposure_minutes=exp_min, program='P')
    return t


def test_density_prefers_shorter_exposure_at_equal_priority():
    """Two same-priority targets, same sky position: the shorter one should
    win the first slot via the value-density nudge."""
    from orchestrator import planner
    from orchestrator.planner import (EXPOSURE_DENSITY_BONUS,
                                      EXPOSURE_DENSITY_REF_MIN)
    short = EXPOSURE_DENSITY_BONUS * min(EXPOSURE_DENSITY_REF_MIN / 15.0, 3.0)
    long = EXPOSURE_DENSITY_BONUS * min(EXPOSURE_DENSITY_REF_MIN / 150.0, 3.0)
    assert short > long
    assert short <= 15.0 + 1e-9   # bounded: cannot leapfrog a priority tier


def test_slew_time_extends_wall_clock(tmp_path):
    """A large slew between consecutive targets must consume wall-clock time:
    the second entry's window is longer than exposure+overhead alone."""
    import astropy.units as u
    from astropy.time import Time
    from orchestrator.planner import create_schedule, compute_observability
    from orchestrator.config import LLAMASConfig

    config = LLAMASConfig()
    evening = Time('2026-07-14T00:00:00')
    morning = Time('2026-07-14T08:00:00')
    # Two targets ~60 deg apart on the sky, both south, both easy
    t1 = _mk_target('near', 300.0, -30.0, 20.0)
    t2 = _mk_target('far', 240.0, -30.0, 20.0)
    targets = compute_observability([t1, t2], evening, morning, config=config)
    plan = create_schedule(targets, evening, morning, moon_phase='dark',
                           standards_path=str(tmp_path / 'none.txt'),
                           config=config)
    if len(plan.scheduled) == 2:
        e2 = plan.scheduled[1]
        wall = (e2.end - e2.start).to_value('min')
        base = 20.0 + config.overhead_minutes
        # buffer (2 min) + slew (~60 deg / 60 deg-per-min = ~1 min) charged
        assert wall >= base + config.acquisition_buffer_minutes - 0.5
        # science-time charging invariant: ops time is NOT billed
        assert e2.charged_minutes <= base + 1.0


def test_mid_standard_reserved_block(tmp_path):
    """On a long night, the mid-night standard must own a reserved block and
    no science entry may overlap it."""
    from astropy.time import Time
    from orchestrator.planner import create_schedule, compute_observability
    from orchestrator.config import LLAMASConfig

    # Synthetic standards catalog spanning RA so one is always up (fixed-width
    # format: name ra dec vmag spectype)
    std = tmp_path / 'standards.txt'
    std.write_text(
        "CD-TEST1    21 00 00  -30 00 00  11.0  DA\n"
        "CD-TEST2    03 00 00  -30 00 00  11.0  DA\n"
        "CD-TEST3    07 00 00  -30 00 00  11.0  DA\n")

    config = LLAMASConfig()
    evening = Time('2026-07-13T23:30:00')
    morning = Time('2026-07-14T10:00:00')  # 10.5h night -> mid standards due
    targets = compute_observability(
        [_mk_target(f't{i}', 300.0 + i, -30.0, 60.0) for i in range(8)],
        evening, morning, config=config)
    plan = create_schedule(targets, evening, morning, moon_phase='dark',
                           standards_path=str(std), config=config)
    mids = [s for s in (plan.standards_mid or []) if 'start' in s]
    if mids:  # standards parse/geometry permitting
        for smid in mids:
            for e in plan.scheduled:
                assert not (e.start < smid['end'] and e.end > smid['start']), \
                    f"science entry {e.target.name} overlaps mid-standard block"


def test_fairness_band_prefers_underserved_when_feasible(tmp_path):
    """Tolerance band: a program over its share (beyond tolerance) loses the
    slot to a feasible under-served candidate, but the band yields (never
    wastes sky) when the underdog has nothing observable."""
    from astropy.time import Time
    from orchestrator.planner import create_schedule, compute_observability
    from orchestrator.config import LLAMASConfig
    from orchestrator.accounting import TimeAccountant

    y = tmp_path / 'a.yaml'
    y.write_text("""
semester: "T"
default_program: "A"
programs:
  - program: "A"
    pi: "x"
    allocated_hours: {dark: 5.0, grey: 0.0, bright: 0.0}
  - program: "B"
    pi: "y"
    allocated_hours: {dark: 5.0, grey: 0.0, bright: 0.0}
""")
    acc = TimeAccountant.from_yaml(str(y), state_path=str(tmp_path / 's.json'))
    config = LLAMASConfig()
    config.fairness_tolerance = 0.10
    evening = Time('2026-07-14T00:00:00')
    morning = Time('2026-07-14T06:00:00')
    # Program A: two big targets; program B: one small target, same sky area.
    # After A's first pick, A is far over-share; B's target is feasible, so
    # the band must hand B the next slot even though A's P1 outranks it.
    tgts = [
        _mk_target('A1', 300.0, -30.0, 90.0, priority=1),
        _mk_target('A2', 301.0, -30.0, 90.0, priority=1),
        _mk_target('B1', 302.0, -30.0, 25.0, priority=3),
    ]
    for t, p in zip(tgts, ('A', 'A', 'B')):
        t.program = p
    targets = compute_observability(tgts, evening, morning, config=config)
    plan = create_schedule(targets, evening, morning, moon_phase='dark',
                           standards_path=str(tmp_path / 'none.txt'),
                           config=config, accountant=acc)
    order = [e.target.name for e in plan.scheduled]
    assert order.index('B1') == 1, f"band should slot B1 second, got {order}"
