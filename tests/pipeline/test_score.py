"""Tests for the PI-approved selection score (2026-08-18): score = P x V(z) x G x U.

Exercises the factor math in core.magellan_planning directly (no pipeline
runs): U (rest-frame deadline urgency), G (spectrum information gain),
V(z) (Hubble-diagram sample density from the target ledger + prior), and
w_lcq (SALT2 color-error light-curve quality).
"""
import json
import math

import numpy as np
import pytest

from config import ScoreConfig
from core.magellan_planning import (
    compute_g_info,
    compute_score_breakdown,
    compute_u_urgency,
    compute_v_z,
    compute_vz_bins,
    compute_w_lcq,
    ledger_redshift_counts,
)

CFG = ScoreConfig()


# ---------------------------------------------------------------------------
# U — urgency (deadline-shaped, rest-frame)
# ---------------------------------------------------------------------------

def test_u_flat_region_including_pre_peak():
    # all pre-peak and the first 5 rest-days post-peak are maximally urgent
    u = compute_u_urgency(np.array([-20.0, -1.0, 0.0, 3.0, 5.0]), config=CFG)
    assert np.allclose(u, 1.0)


def test_u_rest_frame_conversion():
    # 10 observer-days at z=1 is 5 rest-days -> still in the flat region,
    # whereas the same 10 days at z=0 has decayed
    u_hi_z = compute_u_urgency(np.array([10.0]), z=np.array([1.0]), config=CFG)
    u_lo_z = compute_u_urgency(np.array([10.0]), z=np.array([0.0]), config=CFG)
    assert u_hi_z[0] == pytest.approx(1.0)
    assert u_lo_z[0] == pytest.approx(math.exp(-5.0 / 12.0))
    assert u_hi_z[0] > u_lo_z[0]


def test_u_unknown_z_uses_observer_frame():
    u_nan_z = compute_u_urgency(np.array([17.0]), z=np.array([np.nan]), config=CFG)
    u_no_z = compute_u_urgency(np.array([17.0]), config=CFG)
    assert u_nan_z[0] == pytest.approx(u_no_z[0]) == pytest.approx(math.exp(-1.0))


def test_u_decay_and_floor():
    # one e-folding past the flat region
    u = compute_u_urgency(np.array([5.0 + 12.0]), config=CFG)
    assert u[0] == pytest.approx(math.exp(-1.0))
    # very stale objects hit the floor, not zero
    u_old = compute_u_urgency(np.array([500.0]), config=CFG)
    assert u_old[0] == CFG.u_floor
    # monotone non-increasing past the deadline
    u_seq = compute_u_urgency(np.array([6.0, 10.0, 20.0, 40.0]), config=CFG)
    assert np.all(np.diff(u_seq) < 0)


def test_u_nan_delta_t_propagates():
    u = compute_u_urgency(np.array([np.nan]), config=CFG)
    assert np.isnan(u[0])


# ---------------------------------------------------------------------------
# G — information gain
# ---------------------------------------------------------------------------

def test_g_all_four_cases():
    g = compute_g_info(
        tns_type=['SN Ia', 'SN II', None, None],
        z_source=['tns_specz', 'ned_host', 'tns_specz', 'none'],
        config=CFG)
    assert g[0] == CFG.g_both        # typed + spec-z: nearly nothing to gain
    assert g[1] == CFG.g_type_only   # typed, no spec-z
    assert g[2] == CFG.g_z_only      # spec-z, untyped
    assert g[3] == CFG.g_neither     # neither


def test_g_missing_string_spellings_are_untyped():
    # '', 'nan', None, and float NaN all mean "no spectroscopic type"
    g = compute_g_info(tns_type=['', 'nan', None, float('nan')],
                       z_source=['x', 'x', 'x', 'x'], config=CFG)
    assert np.allclose(g, CFG.g_neither)


# ---------------------------------------------------------------------------
# V(z) — Hubble-diagram sample density
# ---------------------------------------------------------------------------

def test_v_cold_start_increases_with_z():
    # ledger empty: the low-z prior alone makes V monotonically increasing
    # with z, with the top bin at exactly 1.0 after normalization
    v_bins = compute_vz_bins(config=CFG)
    assert len(v_bins) == 10
    assert np.all(np.diff(v_bins) >= 0)
    assert v_bins[-1] == pytest.approx(1.0)
    assert np.all(v_bins >= CFG.v_floor)


def test_v_populated_ledger_bin_lowers_that_bin():
    cold = compute_vz_bins(config=CFG)
    counts = np.zeros(10)
    counts[8] = 50.0                          # 50 observed SNe at z ~ 0.40-0.45
    warm = compute_vz_bins(counts, config=CFG)
    # normalization max moves elsewhere; the populated bin drops
    assert warm[8] < cold[8]
    # an untouched high-z bin is unchanged relative to the (new) max
    assert warm[9] == pytest.approx(1.0)


def test_v_per_candidate_and_unknown_z_neutral():
    z = np.array([0.02, 0.47, np.nan])
    v = compute_v_z(z, config=CFG)
    assert v[1] > v[0]                        # empty high-z bin beats saturated low-z
    # unknown z -> median V of the night's known-z candidates
    assert v[2] == pytest.approx(np.median([v[0], v[1]]))


def test_v_all_unknown_uses_default():
    v = compute_v_z(np.array([np.nan, np.nan]), config=CFG)
    assert np.allclose(v, CFG.v_unknown_default)


def test_ledger_counts_from_file_and_missing_file(tmp_path):
    ledger = {"entries": {
        "a": {"required_seconds_history": [
            {"redshift": 0.30}, {"redshift": None}]},   # last non-null = 0.30
        "b": {"required_seconds_history": [{"redshift": 0.31}]},
        "c": {"required_seconds_history": [{"redshift": None}]},  # no z
        "d": {"required_seconds_history": []},
        "e": {"required_seconds_history": [{"redshift": 0.02}]},
    }}
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    counts = ledger_redshift_counts(str(path), config=CFG)
    assert counts[6] == 2.0     # z in [0.30, 0.35): both 0.30 and 0.31
    assert counts[0] == 1.0     # z = 0.02
    assert counts.sum() == 3.0
    # missing file -> zeros (prior only), no crash
    assert ledger_redshift_counts(str(tmp_path / "nope.json"),
                                  config=CFG).sum() == 0.0


# ---------------------------------------------------------------------------
# w_lcq — SALT2 color-error light-curve quality
# ---------------------------------------------------------------------------

def test_w_lcq_good_bad_missing():
    w = compute_w_lcq(np.array([0.0, 0.06, 10.0, np.nan]), config=CFG)
    assert w[0] == pytest.approx(1.0)                 # perfect color
    assert w[1] == pytest.approx(0.5)                 # c_err at the reference
    assert w[2] == CFG.lcq_floor                      # terrible color: floored
    assert w[3] == pytest.approx(1.0)                 # missing -> neutral


# ---------------------------------------------------------------------------
# score = P x V x G x U (assembled)
# ---------------------------------------------------------------------------

def test_score_breakdown_assembly_and_nan_propagation():
    sb = compute_score_breakdown(
        w_prob=np.array([0.9, 0.9, np.nan]),
        w_iaspec=np.array([1.2, 1.2, 1.0]),
        salt_c_err=np.array([0.03, np.nan, 0.03]),
        tns_type=['SN Ia', None, None],
        z_source=['tns_specz', 'none', 'none'],
        delta_t=np.array([2.0, 2.0, 2.0]),
        z=np.array([0.05, 0.05, 0.05]),
        config=CFG)
    # every factor multiplies through
    assert sb['score'][0] == pytest.approx(
        sb['p_usable'][0] * sb['v_z'][0] * sb['g_info'][0] * sb['u_urgency'][0])
    # spec-typed + spec-z'd object is heavily deprioritized vs the untyped one
    assert sb['g_info'][0] == CFG.g_both and sb['g_info'][1] == CFG.g_neither
    assert sb['score'][0] < sb['score'][1]
    # NaN w_prob propagates to NaN score (merit convention)
    assert np.isnan(sb['score'][2])
