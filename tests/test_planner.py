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
