"""Tests for the LLAMAS exposure cascade (chunk F: R12).

The redshift branch of estimate_llamas_exposure now linearly interpolates the
proposal Table 1 (z -> minutes) instead of stepping between rows — a hair of
redshift no longer swings the estimate by ~85 min. Fully in-memory: the table
lives in LLAMASConfig; no file/network access.
"""

import math

from orchestrator.config import LLAMAS_CONFIG
from orchestrator.normalize import estimate_llamas_exposure


def _table_point(z):
    """Return the tabulated (max_z, minutes) row at exactly ``z``, if any."""
    for max_z, exp_min, _ in LLAMAS_CONFIG.exposure_table:
        if abs(max_z - z) < 1e-9:
            return exp_min
    return None


def test_exposure_interpolates_between_rows():
    """A z strictly between two table rows yields a value strictly between the
    two rows' minutes (no cliff)."""
    # Rows include (0.30, 95) and (0.35, 100). z=0.305 sits just past 0.30.
    lo = _table_point(0.30)
    hi = _table_point(0.35)
    assert lo is not None and hi is not None

    exp_lo, _ = estimate_llamas_exposure(0.30, float('nan'))
    exp_hi, _ = estimate_llamas_exposure(0.35, float('nan'))
    exp_mid, _ = estimate_llamas_exposure(0.325, float('nan'))

    assert exp_lo == lo
    assert exp_hi == hi
    # Midpoint must land strictly between the bracketing rows.
    assert exp_lo < exp_mid < exp_hi


def test_exposure_no_cliff_near_boundary():
    """A hair of redshift past a row moves the estimate only a little, not by
    the full inter-row jump (regression on the ~85 min cliff)."""
    exp_at, _ = estimate_llamas_exposure(0.30, float('nan'))
    exp_just_past, _ = estimate_llamas_exposure(0.301, float('nan'))
    # Difference is a small fraction of the row-to-row gap (95 -> 100 over
    # z 0.30 -> 0.35), nowhere near a cliff.
    assert abs(exp_just_past - exp_at) < 1.0


def test_exposure_clamps_below_first_row():
    """Below the lowest tabulated z, clamp to the first row's value."""
    first_z, first_min, _ = sorted(LLAMAS_CONFIG.exposure_table)[0]
    exp, _ = estimate_llamas_exposure(first_z - 0.1, float('nan'))
    assert exp == first_min


def test_exposure_clamps_above_last_row():
    """Above the highest tabulated z, clamp to the last row's value (no
    runaway extrapolation)."""
    last_z, last_min, _ = sorted(LLAMAS_CONFIG.exposure_table)[-1]
    exp, _ = estimate_llamas_exposure(last_z + 0.5, float('nan'))
    assert exp == last_min


def test_magnitude_fallback_branch_still_works():
    """With no redshift, the magnitude-scaling branch is used (mag 20 -> 45)."""
    exp, constraint = estimate_llamas_exposure(float('nan'), 20.0, moon='grey')
    assert math.isclose(exp, 45.0, rel_tol=1e-6)
    assert constraint == 'grey'

    # Fainter target -> longer (but bounded) exposure.
    exp_faint, _ = estimate_llamas_exposure(float('nan'), 21.0)
    assert exp_faint > exp


def test_fixed_fallback_when_nothing_known():
    """No redshift and no magnitude -> the fixed fallback."""
    exp, constraint = estimate_llamas_exposure(float('nan'), float('nan'))
    assert exp == LLAMAS_CONFIG.fallback_exposure_minutes
    assert constraint == 'any'
