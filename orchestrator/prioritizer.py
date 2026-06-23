"""Composite priority scoring for LLAMAS target scheduling.

Combines science priority, observability, time budget, light curve phase,
and keyword signals into a single score for the greedy scheduler.
"""

import logging
import math
from typing import Optional, Tuple

from astropy.time import Time
import astropy.units as u

from .models import Target
from .accounting import TimeAccountant
from .config import LLAMASConfig, LLAMAS_CONFIG

logger = logging.getLogger(__name__)

# Priority → base science weight
SCIENCE_WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.2}

# ---------------------------------------------------------------------------
# Composite-score scaling constants
# ---------------------------------------------------------------------------
# The composite score is a *multiplicative science core* plus two *additive
# nudges*. Each constant below sets the relative weight of one term; they are
# chosen so a PI can reconstruct any ranking by hand from the breakdown dict.
#
#   science_term       = SCIENCE_SCALE × science × budget × phase
#   observability_term = OBSERVABILITY_BONUS × observability
#   keyword_term       = KEYWORD_SCALE × keyword_adj
#   total              = science_term + observability_term + keyword_term
#
# All three core factors (science, budget, phase) live in [0, 1] (phase is
# clamped below), so the science term spans [0, SCIENCE_SCALE]. Setting
# SCIENCE_SCALE = 100 makes the science core the dominant axis: a P1 target on
# a healthy budget at peak (1×1×1) earns ~100, while a P4 (0.2) earns ~20.
SCIENCE_SCALE = 100.0

# Observability is a fraction in [0, 1] of the night the target is up. It is an
# additive *nudge* (bonus), not a multiplier, so a poorly-observable but
# scientifically vital target is not zeroed out — it just loses up to 20 pts.
# At 20, observability can swing a ranking between near-equal science tiers
# (e.g. it can lift a fully-observable P2 core above a barely-up P1 core only
# when the science gap is small), but cannot by itself outrank a full priority
# tier (which differ by ~30 pts: 100→70→40→20).
OBSERVABILITY_BONUS = 20.0

# Keyword adjustment is a net signed value (boosts − demotions, typically in
# roughly [-0.6, +0.6]). Scaled by 50 it becomes a meaningful but bounded human
# override: an "urgent"/"high priority" note (~+0.3..+0.5) adds ~15-25 pts, and
# a "backup"/"filler" note subtracts similarly — enough to reorder within a
# tier or nudge across an adjacent one, never to dominate the science core.
KEYWORD_SCALE = 50.0

# Keyword adjustments: scan target notes for scheduling signals
KEYWORD_BOOSTS = {
    'high priority': 0.3,
    'urgent': 0.2,
    'too': 0.25,        # target of opportunity
    'classification needed': 0.2,
    'rising': 0.15,
    'near peak': 0.15,
    'precursor': 0.1,
}
KEYWORD_DEMOTIONS = {
    'backup': -0.3,
    'filler': -0.2,
    'low priority': -0.3,
    'if time': -0.2,
}


def _keyword_adjustment(notes: str) -> float:
    """Scan notes for scheduling keywords, return net adjustment."""
    if not notes:
        return 0.0

    notes_lower = notes.lower()
    adj = 0.0
    for kw, boost in KEYWORD_BOOSTS.items():
        if kw in notes_lower:
            adj += boost
    for kw, demote in KEYWORD_DEMOTIONS.items():
        if kw in notes_lower:
            adj += demote
    return adj


def _clamp01(x: float) -> float:
    """Clamp a value to [0, 1]; non-finite → 0.0."""
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _phase_factor(target: Target, phase_preference: str = 'peak',
                  config: LLAMASConfig = None) -> float:
    """Light-curve phase factor for a program's desired phase.

    Different MAGNETS programs want SNe at different light-curve phases:
    cosmology/standardization wants PEAK (best S/N, the SALT epoch); progenitor/
    CSM/exotic science wants the RISE (early ejecta, flash spectroscopy). The
    factor is a Gaussian centered on the program's PREFERRED time-from-peak:

        factor = exp(-(delta_t - dt_pref)² / (2·τ²))

    where ``dt_pref`` = config.phase_preference_offsets[phase_preference] (days;
    0 for 'peak', -7 for 'rising') and τ = config.phase_tau_days. It is maximal
    (1.0) when the target's signed delta_t equals the preferred offset and falls
    off symmetrically either side.

    Cascade:
    1. If ``target.delta_t`` is finite -> the signed, preference-aware Gaussian.
    2. Else if ``target.phase_weight`` is finite -> legacy symmetric weight
       (peak-meaningful only; preference is ignored because the sign is lost).
    3. Else -> 1.0 (neutral; do NOT fabricate near-peak).

    Clamped to [0, 1] so a malformed/over-unity weight can never let a stale
    near-peak target blow past the science core (R2): the multiplicative core
    is bounded at 1.0 and the science term at SCIENCE_SCALE.
    """
    if config is None:
        config = LLAMAS_CONFIG
    if math.isfinite(target.delta_t):
        dt_pref = config.phase_preference_offsets.get(phase_preference, 0.0)
        tau = config.phase_tau_days
        return _clamp01(math.exp(
            -((target.delta_t - dt_pref) ** 2) / (2.0 * tau ** 2)))
    if math.isfinite(target.phase_weight):
        return _clamp01(target.phase_weight)
    return 1.0


def compute_composite_score(target: Target,
                            accountant: Optional[TimeAccountant] = None,
                            evening: Optional[Time] = None,
                            morning: Optional[Time] = None,
                            config: LLAMASConfig = None,
                            moon_phase: Optional[str] = None,
                            completeness: float = 1.0
                            ) -> Tuple[float, dict]:
    """Compute composite priority score for scheduling.

    Scoring model
    -------------
    The score is a **multiplicative science core** plus two **additive
    nudges**, so a PI can reconstruct any ranking by hand from the returned
    breakdown:

        science_term       = SCIENCE_SCALE × science × budget × phase × completeness
        observability_term = OBSERVABILITY_BONUS × observability
        keyword_term       = KEYWORD_SCALE × keyword_adj
        total              = science_term + observability_term + keyword_term

    The four core factors (science, budget, phase, completeness) all live in
    [0, 1] (phase is clamped here, observability is clamped, budget/science are
    sane by construction, completeness comes from the per-target ledger), so
    the multiplicative core is bounded at 1.0 and the science term at
    SCIENCE_SCALE. ``completeness`` is the per-target integration ledger factor
    (W11): 1.0 for a fresh target, → 0.0 as it accumulates enough integration
    time, so a finished target drops to the bottom of the ranking even if its
    science priority is high. Observability and keywords are *additive* nudges —
    they reorder within or across adjacent tiers but cannot, by themselves,
    leapfrog the science core (see the constant docstrings above for why the
    scales are 100 / 20 / 50).

    Components
    ----------
    science_weight : float
        P1-P4 mapped to 1.0/0.7/0.4/0.2 (in [0, 1]).
    budget_factor : float
        1.0/0.5/0.1 based on remaining program hours (in [0, 1]).
    observability : float
        Fraction of night target is above airmass limit, clamped to [0, 1].
    phase_factor : float
        Light curve phase weight, clamped to [0, 1] — near-peak targets score
        higher but can never exceed 1.0 (R2).
    keyword_adj : float
        Net boost/demotion from notes keywords (signed).
    completeness : float
        Per-target integration completeness factor in [0, 1] from the W11
        ledger; 1.0 when no ledger is in use (neutral).

    Returns
    -------
    (score, breakdown) : tuple of (float, dict)
        ``score`` is the composite total (higher = schedule first).
        ``breakdown`` records every input factor and term:
        ``science``, ``budget``, ``phase``, ``completeness``,
        ``observability``, ``keyword_adj``, ``science_term``,
        ``observability_term``, ``keyword_term``, ``total``.
    """
    if config is None:
        config = LLAMAS_CONFIG

    # Science priority (table values are already in [0, 1])
    science = SCIENCE_WEIGHTS.get(target.priority, 0.3)

    # Budget factor (accountant returns 1.0/0.5/0.1, all in [0, 1]); clamp
    # defensively in case a future accountant returns out-of-range values.
    budget = 1.0
    if accountant is not None:
        budget = accountant.get_budget_factor(target.program, moon_phase)
    budget = _clamp01(budget)

    # Observability fraction, clamped to [0, 1].
    observability = 0.5  # default if twilight not provided
    if evening is not None and morning is not None:
        night_hours = (morning - evening).to(u.hour).value
        if night_hours > 0 and target.window_hours > 0:
            observability = target.window_hours / night_hours
    observability = _clamp01(observability)

    # Phase factor (clamped to [0, 1] inside _phase_factor). The program's
    # phase preference (peak vs rising) selects the Gaussian center; default
    # 'peak' when no accountant is supplied to look it up.
    phase_preference = 'peak'
    if accountant is not None:
        phase_preference = accountant.get_phase_preference(target.program)
    phase = _phase_factor(target, phase_preference, config)

    # Completeness factor (W11): per-target integration ledger weight, clamped
    # to [0, 1]. 1.0 (neutral) when no ledger is in use.
    completeness = _clamp01(completeness)

    # Keyword adjustment (signed; not clamped — it is a bounded human override)
    kw_adj = _keyword_adjustment(target.notes)

    # Composite: multiplicative science core + additive observability/keyword.
    science_term = SCIENCE_SCALE * science * budget * phase * completeness
    observability_term = OBSERVABILITY_BONUS * observability
    keyword_term = KEYWORD_SCALE * kw_adj
    score = science_term + observability_term + keyword_term

    breakdown = {
        'science': science,
        'budget': budget,
        'phase': phase,
        # Label only (which Gaussian center produced ``phase``). Does NOT affect
        # the total == sum-of-terms invariant since phase is folded into
        # science_term.
        'phase_preference': phase_preference,
        'completeness': completeness,
        'observability': observability,
        'keyword_adj': kw_adj,
        'science_term': science_term,
        'observability_term': observability_term,
        'keyword_term': keyword_term,
        'total': score,
    }
    return score, breakdown


def rank_targets(targets: list,
                 accountant: Optional[TimeAccountant] = None,
                 evening: Optional[Time] = None,
                 morning: Optional[Time] = None,
                 config: LLAMASConfig = None,
                 moon_phase: Optional[str] = None,
                 ledger=None,
                 required_minutes_by_target: Optional[dict] = None
                 ) -> Tuple[dict, dict]:
    """Score all targets and return (scores, breakdowns) mappings.

    Also stores the composite score in each target's merit_score field and the
    per-component breakdown in each target's score_breakdown field (so the
    summary writer can persist/display it, mirroring the alert-pipeline merit
    breakdown — R14).

    Parameters
    ----------
    ledger : TargetLedger, optional
        Per-target integration ledger (W11). When provided, each target's
        completeness factor is computed from its cumulative integration vs its
        full required time and folded into the composite science core. When
        None, completeness is 1.0 (neutral) and behaviour matches the prior
        ledger-less scoring exactly.
    required_minutes_by_target : dict, optional
        {target.name: full_required_minutes} used with ``ledger`` to compute
        completeness. Falls back to ``target.required_minutes_full`` / 0 when a
        target is missing here.

    Returns
    -------
    (scores, breakdowns) : tuple of (dict, dict)
        ``scores``: {target.name: composite_score} for create_schedule().
        ``breakdowns``: {target.name: breakdown_dict} from
        compute_composite_score, for write_summary / score_breakdown.json.
    """
    scores = {}
    breakdowns = {}
    required_minutes_by_target = required_minutes_by_target or {}
    for t in targets:
        completeness = 1.0
        if ledger is not None:
            req_min = required_minutes_by_target.get(
                t.name, getattr(t, 'required_minutes_full', float('nan')))
            if req_min is None or not math.isfinite(req_min):
                req_min = 0.0
            completeness = ledger.completeness_factor(t, req_min)
        s, breakdown = compute_composite_score(
            t, accountant, evening, morning, config, moon_phase,
            completeness=completeness)
        t.merit_score = s
        t.score_breakdown = breakdown
        scores[t.name] = s
        breakdowns[t.name] = breakdown

    ranked = sorted(targets, key=lambda t: t.merit_score, reverse=True)
    for i, t in enumerate(ranked):
        bd = t.score_breakdown or {}
        pref = bd.get('phase_preference', 'peak')
        logger.debug("Rank %2d: %-18s score=%.1f P%d budget=%.1f phase=%.2f "
                     "(%s) complete=%.2f",
                     i + 1, t.name, t.merit_score, t.priority,
                     accountant.get_budget_factor(t.program, moon_phase)
                     if accountant else 1.0,
                     bd.get('phase', _phase_factor(t, pref, config)),
                     pref, bd.get('completeness', 1.0))

    return scores, breakdowns
