"""Exposures are sized at the mag AT OBSERVATION, not at the light-curve peak.

A SN scheduled off peak is fainter than peak_mag; sizing at peak under-exposes
it. The pipeline path carries delta_t (phase at the observation night); the
manual path (delta_t NaN) is unaffected.
"""
import math

from orchestrator.normalize import (predicted_mag_at_observation,
                                     estimate_llamas_exposure)
from orchestrator.config import LLAMASConfig

CFG = LLAMASConfig()


def test_peak_used_when_delta_t_unknown():
    # manual path: mag is already the anticipated value, delta_t is NaN
    assert predicted_mag_at_observation(19.0, float("nan"), CFG) == 19.0


def test_declining_is_fainter_than_peak():
    m = predicted_mag_at_observation(19.0, 20.0, CFG)   # +20 d post-peak
    assert m == 19.0 + min(20 * CFG.mag_decline_per_day, CFG.mag_fade_cap)
    assert m > 19.0


def test_rising_is_also_fainter_than_peak():
    assert predicted_mag_at_observation(19.0, -10.0, CFG) > 19.0


def test_fade_is_capped():
    m = predicted_mag_at_observation(19.0, 1000.0, CFG)
    assert m == 19.0 + CFG.mag_fade_cap


def test_nan_peak_returns_nan():
    assert math.isnan(predicted_mag_at_observation(float("nan"), 5.0, CFG))


def test_declining_target_gets_longer_exposure():
    # Base mag chosen faint enough that the exposure is above the 10-min floor,
    # so the fade (fainter at +20 d) shows up as a longer integration.
    at_peak, _ = estimate_llamas_exposure(float("nan"), 22.5, "dark", delta_t=0.0)
    declining, _ = estimate_llamas_exposure(float("nan"), 22.5, "dark", delta_t=20.0)
    assert declining > at_peak     # fainter at +20 d -> longer integration
