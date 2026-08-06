"""Tests for per-program light-curve phase preference (peak vs rising).

All offline: synthetic Targets, temp-file CSV/YAML only. Nothing contacts a
broker, database, or external API.
"""

import math

import pytest

from orchestrator.models import Target
from orchestrator.config import LLAMAS_CONFIG
from orchestrator.accounting import TimeAccountant
from orchestrator.normalize import load_from_rubinalerts, load_targets_csv
from orchestrator.prioritizer import (
    _phase_factor,
    compute_composite_score,
)


# ---------------------------------------------------------------------------
# _phase_factor: preference-aware Gaussian
# ---------------------------------------------------------------------------

def test_phase_factor_peak_maximal_at_zero():
    """A 'peak' program's factor is maximal at delta_t == 0 and falls off."""
    at_peak = Target(name='peak', ra_deg=150.0, dec_deg=2.0, delta_t=0.0)
    pre = Target(name='pre', ra_deg=150.0, dec_deg=2.0, delta_t=-7.0)
    post = Target(name='post', ra_deg=150.0, dec_deg=2.0, delta_t=7.0)

    f0 = _phase_factor(at_peak, 'peak')
    assert f0 == pytest.approx(1.0)
    assert _phase_factor(pre, 'peak') < f0
    assert _phase_factor(post, 'peak') < f0
    # Symmetric falloff around the preferred offset (dt_pref=0).
    assert _phase_factor(pre, 'peak') == pytest.approx(
        _phase_factor(post, 'peak'))


def test_phase_factor_rising_maximal_at_offset():
    """A 'rising' program's factor is maximal at delta_t == dt_pref (-7 d)."""
    dt_pref = LLAMAS_CONFIG.phase_preference_offsets['rising']
    at_pref = Target(name='atpref', ra_deg=150.0, dec_deg=2.0, delta_t=dt_pref)
    one_side = Target(name='hi', ra_deg=150.0, dec_deg=2.0, delta_t=dt_pref + 4)
    other = Target(name='lo', ra_deg=150.0, dec_deg=2.0, delta_t=dt_pref - 4)

    fpref = _phase_factor(at_pref, 'rising')
    assert fpref == pytest.approx(1.0)
    assert _phase_factor(one_side, 'rising') < fpref
    # Symmetric falloff around dt_pref.
    assert _phase_factor(one_side, 'rising') == pytest.approx(
        _phase_factor(other, 'rising'))


# ---------------------------------------------------------------------------
# Program preference flips the ordering of a pre-peak vs at-peak target
# ---------------------------------------------------------------------------

def test_rising_program_prefers_pre_peak(sample_allocations_path):
    """A rising-preference program scores a pre-peak target (delta_t=-7)
    STRICTLY higher than a peak target (delta_t=0); a peak-preference program
    does the reverse."""
    accountant = TimeAccountant.from_yaml(sample_allocations_path)
    # Villar = rising, Stubbs = peak (per the example YAML).
    rising_prog = 'MAGNETS-Villar'
    peak_prog = 'MAGNETS-Stubbs'

    def score(program, delta_t):
        t = Target(name='t', ra_deg=150.0, dec_deg=2.0, priority=1,
                   program=program, delta_t=delta_t)
        s, _ = compute_composite_score(t, accountant=accountant)
        return s

    # Rising program: pre-peak beats at-peak.
    assert score(rising_prog, -7.0) > score(rising_prog, 0.0)
    # Peak program: at-peak beats pre-peak.
    assert score(peak_prog, 0.0) > score(peak_prog, -7.0)


# ---------------------------------------------------------------------------
# delta_t threading from CSV loaders
# ---------------------------------------------------------------------------

def test_delta_t_from_rubinalerts_csv(tmp_path):
    """A candidates.csv with a delta_t column threads it onto the Target."""
    csv = tmp_path / 'candidates.csv'
    csv.write_text(
        "object_id,ra,dec,merit,w_time,delta_t\n"
        "OBJ1,150.0,2.0,0.9,0.8,-7.0\n"
        "OBJ2,151.0,2.0,0.5,0.6,3.0\n"
    )
    targets = load_from_rubinalerts(str(csv))
    by_name = {t.name: t for t in targets}
    assert by_name['OBJ1'].delta_t == pytest.approx(-7.0)
    assert by_name['OBJ2'].delta_t == pytest.approx(3.0)


def test_delta_t_from_peak_and_night_mjd(tmp_path):
    """A manual CSV with peak_mjd + a supplied night_mjd derives
    delta_t = night - peak."""
    csv = tmp_path / 'targets.csv'
    csv.write_text(
        "name,ra,dec,peak_mjd\n"
        "MAN1,150.0,2.0,61100.0\n"
    )
    targets = load_targets_csv(str(csv), night_mjd=61093.0)
    assert targets[0].delta_t == pytest.approx(-7.0)


def test_explicit_delta_t_column_wins(tmp_path):
    """An explicit delta_t column overrides the peak_mjd derivation."""
    csv = tmp_path / 'targets.csv'
    csv.write_text(
        "name,ra,dec,peak_mjd,delta_t\n"
        "MAN1,150.0,2.0,61100.0,2.5\n"
    )
    targets = load_targets_csv(str(csv), night_mjd=61093.0)
    assert targets[0].delta_t == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Backward compatibility: phase_weight-only target, default 'peak'
# ---------------------------------------------------------------------------

def test_phase_weight_only_legacy_behavior():
    """A target with only phase_weight (no delta_t) under default 'peak'
    behaves exactly as the legacy symmetric weight."""
    t = Target(name='legacy', ra_deg=150.0, dec_deg=2.0, priority=2,
               phase_weight=0.8)
    assert not math.isfinite(t.delta_t)
    assert _phase_factor(t, 'peak') == pytest.approx(0.8)
    # And the rising preference cannot change it (sign is lost without delta_t).
    assert _phase_factor(t, 'rising') == pytest.approx(0.8)


def test_breakdown_invariant_holds_with_phase_preference():
    """total == science_term + observability_term + keyword_term still holds;
    phase_preference is a label only."""
    t = Target(name='T', ra_deg=150.0, dec_deg=2.0, priority=1,
               delta_t=-7.0, keywords=['high_priority'])
    s, bd = compute_composite_score(t)
    assert 'phase_preference' in bd
    assert bd['total'] == pytest.approx(
        bd['science_term'] + bd['observability_term'] + bd['keyword_term'])


# ---------------------------------------------------------------------------
# get_phase_preference
# ---------------------------------------------------------------------------

def test_get_phase_preference_from_yaml(sample_allocations_path):
    """Returns the YAML value; defaults to 'peak' for unknown programs."""
    accountant = TimeAccountant.from_yaml(sample_allocations_path)
    assert accountant.get_phase_preference('MAGNETS-Stubbs') == 'peak'
    assert accountant.get_phase_preference('MAGNETS-Villar') == 'rising'
    assert accountant.get_phase_preference('NOT-A-PROGRAM') == 'peak'
