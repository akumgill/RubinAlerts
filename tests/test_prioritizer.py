"""Tests for composite-score transparency (chunk E: R2, R14).

All in-memory: Target objects built directly, no accountant/network needed.
"""

import math

from orchestrator.models import Target
from orchestrator.prioritizer import (
    compute_composite_score,
    rank_targets,
    SCIENCE_SCALE,
    SCIENCE_WEIGHTS,
    OBSERVABILITY_BONUS,
    KEYWORD_SCALE,
)


def test_returns_score_and_breakdown_tuple():
    """compute_composite_score returns (float, dict) with all keys."""
    t = Target(name='T1', ra_deg=150.0, dec_deg=2.0, priority=1)
    result = compute_composite_score(t)
    assert isinstance(result, tuple) and len(result) == 2
    score, breakdown = result
    assert isinstance(score, float)
    for key in ('science', 'budget', 'phase', 'observability', 'keyword_adj',
                'science_term', 'observability_term', 'keyword_term', 'total'):
        assert key in breakdown


def test_phase_weight_above_one_clamps():
    """phase_weight > 1 (e.g. 1.5) clamps to 1.0 in the breakdown (R2)."""
    t = Target(name='T1', ra_deg=150.0, dec_deg=2.0, priority=1,
               phase_weight=1.5)
    _, bd = compute_composite_score(t)
    assert bd['phase'] == 1.0


def test_breakdown_components_combine_to_score():
    """Each term equals its formula and they sum to the returned score."""
    t = Target(name='T1', ra_deg=150.0, dec_deg=2.0, priority=2,
               phase_weight=0.8, notes='urgent classification needed')
    score, bd = compute_composite_score(t)

    assert math.isclose(
        bd['science_term'],
        SCIENCE_SCALE * bd['science'] * bd['budget'] * bd['phase'])
    assert math.isclose(
        bd['observability_term'], OBSERVABILITY_BONUS * bd['observability'])
    assert math.isclose(bd['keyword_term'], KEYWORD_SCALE * bd['keyword_adj'])
    assert math.isclose(
        score,
        bd['science_term'] + bd['observability_term'] + bd['keyword_term'])
    assert math.isclose(bd['total'], score)


def test_science_term_bounded_by_scale():
    """With all core factors in [0,1], the science term cannot exceed SCALE."""
    t = Target(name='T1', ra_deg=150.0, dec_deg=2.0, priority=1,
               phase_weight=999.0)  # absurd phase, must clamp
    _, bd = compute_composite_score(t)
    assert bd['science_term'] <= SCIENCE_SCALE + 1e-9


def test_stale_p1_does_not_beat_fresh_p4_when_clamped():
    """A stale P1 (low phase) must NOT outrank a fresh P4 only via an
    unbounded phase factor. With clamping, phase in [0,1] keeps the P1 science
    core ordered sensibly: a P1 at full phase beats a fresh P4 at full phase.

    The regression guarded here: an over-unity phase_weight on a stale P1 used
    to inflate its score past intent. After clamping, a P1 with a *low* phase
    can legitimately fall below a P4 at peak — and the science cores reflect
    that honestly rather than via runaway phase inflation.
    """
    # Stale P1: low phase weight -> science core shrinks.
    p1_stale = Target(name='P1_stale', ra_deg=150.0, dec_deg=2.0,
                      priority=1, phase_weight=0.1)
    # Fresh P4: at peak.
    p4_fresh = Target(name='P4_fresh', ra_deg=150.0, dec_deg=2.0,
                      priority=4, phase_weight=1.0)

    s1, bd1 = compute_composite_score(p1_stale)
    s4, bd4 = compute_composite_score(p4_fresh)

    # Both phase factors are in [0,1]; neither exceeds 1.0.
    assert 0.0 <= bd1['phase'] <= 1.0
    assert 0.0 <= bd4['phase'] <= 1.0

    # Science core arithmetic, fully reconstructable by hand:
    #   P1_stale science_term = 100 * 1.0 * 1.0 * 0.1 = 10.0
    #   P4_fresh science_term = 100 * 0.2 * 1.0 * 1.0 = 20.0
    assert math.isclose(bd1['science_term'],
                        SCIENCE_SCALE * SCIENCE_WEIGHTS[1] * 1.0 * 0.1)
    assert math.isclose(bd4['science_term'],
                        SCIENCE_SCALE * SCIENCE_WEIGHTS[4] * 1.0 * 1.0)
    # At the chosen constants the fresh P4 outranks the stale P1.
    assert s4 > s1


def test_full_p1_beats_full_p4():
    """A P1 at peak outranks a P4 at peak (sanity on tier ordering)."""
    p1 = Target(name='P1', ra_deg=150.0, dec_deg=2.0, priority=1,
                phase_weight=1.0)
    p4 = Target(name='P4', ra_deg=150.0, dec_deg=2.0, priority=4,
                phase_weight=1.0)
    s1, _ = compute_composite_score(p1)
    s4, _ = compute_composite_score(p4)
    assert s1 > s4


def test_rank_targets_returns_scores_and_breakdowns():
    """rank_targets returns (scores, breakdowns) and attaches per-target."""
    targets = [
        Target(name='A', ra_deg=150.0, dec_deg=2.0, priority=1),
        Target(name='B', ra_deg=151.0, dec_deg=2.0, priority=3),
    ]
    scores, breakdowns = rank_targets(targets)
    assert set(scores) == {'A', 'B'}
    assert set(breakdowns) == {'A', 'B'}
    for t in targets:
        assert t.score_breakdown is not None
        assert math.isclose(t.merit_score, t.score_breakdown['total'])
