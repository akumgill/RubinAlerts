"""Tests for the redshift-preference weight w_z (source differentiator)."""
import math

from orchestrator.config import LLAMASConfig
from orchestrator.models import Target
from orchestrator.prioritizer import _redshift_factor, compute_composite_score


def _t(z):
    return Target(name='t', ra_deg=150.0, dec_deg=-20.0, redshift=z)


def test_peaks_inside_preferred_window():
    cfg = LLAMASConfig()  # range (0.6, 0.8)
    assert _redshift_factor(_t(0.6), cfg) == 1.0
    assert _redshift_factor(_t(0.7), cfg) == 1.0
    assert _redshift_factor(_t(0.8), cfg) == 1.0


def test_tapers_and_floors_outside_window():
    cfg = LLAMASConfig()
    near = _redshift_factor(_t(0.5), cfg)     # 0.1 below window
    far = _redshift_factor(_t(0.1), cfg)      # nearby SN, deep in downtime range
    assert cfg.z_preference_floor <= near < 1.0
    assert near > far or math.isclose(far, cfg.z_preference_floor)
    # far-off nearby SNe hit the floor, not zero
    assert far == cfg.z_preference_floor


def test_unknown_redshift_is_neutral():
    cfg = LLAMASConfig()
    assert _redshift_factor(_t(float('nan')), cfg) == 1.0


def test_disabled_is_neutral():
    cfg = LLAMASConfig(z_preference_enabled=False)
    assert _redshift_factor(_t(0.1), cfg) == 1.0


def test_high_z_outranks_nearby_in_composite():
    cfg = LLAMASConfig()
    # identical targets except redshift; high-z (in window) must score higher.
    hi, _ = compute_composite_score(_t(0.7), config=cfg)
    lo, _ = compute_composite_score(_t(0.08), config=cfg)
    assert hi > lo
    # and the breakdown exposes the factor
    _, bd = compute_composite_score(_t(0.7), config=cfg)
    assert bd['redshift_pref'] == 1.0
