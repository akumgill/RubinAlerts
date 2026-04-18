"""Composite priority scoring for LLAMAS target scheduling.

Combines science priority, observability, time budget, light curve phase,
and keyword signals into a single score for the greedy scheduler.
"""

import logging
import math
from typing import Optional

from astropy.time import Time
import astropy.units as u

from .models import Target
from .accounting import TimeAccountant
from .config import LLAMASConfig, LLAMAS_CONFIG

logger = logging.getLogger(__name__)

# Priority → base science weight
SCIENCE_WEIGHTS = {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.2}

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


def _phase_factor(target: Target) -> float:
    """Light curve phase factor: targets near peak are more valuable.

    Uses phase_weight from the alert pipeline (w_time = exp(-dt²/2τ²))
    if available. Falls back to 1.0 for manually-entered targets or
    those without timing info.
    """
    if math.isfinite(target.phase_weight):
        return target.phase_weight
    return 1.0


def compute_composite_score(target: Target,
                            accountant: Optional[TimeAccountant] = None,
                            evening: Optional[Time] = None,
                            morning: Optional[Time] = None,
                            config: LLAMASConfig = None) -> float:
    """Compute composite priority score for scheduling.

    Components
    ----------
    science_weight : float
        P1-P4 mapped to 1.0/0.7/0.4/0.2.
    budget_factor : float
        1.0/0.5/0.1 based on remaining program hours.
    observability : float
        Fraction of night target is above airmass limit (0-1).
    phase_factor : float
        Light curve phase weight — near-peak targets score higher.
    keyword_adj : float
        Boost/demotion from notes keywords.

    Returns
    -------
    float
        Composite score; higher = schedule first.
    """
    if config is None:
        config = LLAMAS_CONFIG

    # Science priority
    science = SCIENCE_WEIGHTS.get(target.priority, 0.3)

    # Budget factor
    budget = 1.0
    if accountant is not None:
        budget = accountant.get_budget_factor(target.program)

    # Observability fraction
    observability = 0.5  # default if twilight not provided
    if evening is not None and morning is not None:
        night_hours = (morning - evening).to(u.hour).value
        if night_hours > 0 and target.window_hours > 0:
            observability = min(1.0, target.window_hours / night_hours)

    # Phase factor
    phase = _phase_factor(target)

    # Keyword adjustment
    kw_adj = _keyword_adjustment(target.notes)

    # Composite: science × budget × phase as base, observability and keywords additive
    score = science * 100.0 * budget * phase + observability * 20.0 + kw_adj * 50.0

    return score


def rank_targets(targets: list,
                 accountant: Optional[TimeAccountant] = None,
                 evening: Optional[Time] = None,
                 morning: Optional[Time] = None,
                 config: LLAMASConfig = None) -> dict:
    """Score all targets and return name → score mapping.

    Also stores the composite score in each target's merit_score field.

    Returns
    -------
    dict
        {target.name: composite_score} for use in create_schedule().
    """
    scores = {}
    for t in targets:
        s = compute_composite_score(t, accountant, evening, morning, config)
        t.merit_score = s
        scores[t.name] = s

    ranked = sorted(targets, key=lambda t: t.merit_score, reverse=True)
    for i, t in enumerate(ranked):
        logger.debug("Rank %2d: %-18s score=%.1f P%d budget=%.1f phase=%.2f",
                     i + 1, t.name, t.merit_score, t.priority,
                     accountant.get_budget_factor(t.program) if accountant else 1.0,
                     _phase_factor(t))

    return scores
