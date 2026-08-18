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
    compute_e_east,
    compute_g_info,
    compute_score_breakdown,
    compute_u_urgency,
    compute_v_z,
    compute_vz_bins,
    compute_w_lcq,
    evening_twilight_lst_hours,
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
    # Pin the Chris 2026-08-18 decision: an already-typed object without a
    # spec-z gains LITTLE from a live spectrum (host z comes later via MOS) —
    # 0.15, down from the proposal's 0.7.
    assert CFG.g_type_only == 0.15
    # an untyped spec-z object still gains most of a slot (the type)
    assert CFG.g_z_only > CFG.g_type_only


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
# E — east-rising observability longevity (stamped #4)
# ---------------------------------------------------------------------------

def test_e_east_of_meridian_full_weight():
    # LST 6 h; RA east of the meridian (larger RA = smaller HA) -> rising, 1.0
    lst = 6.0
    e = compute_e_east(np.array([6.0 * 15 + 30.0, 6.0 * 15 + 90.0]), lst, CFG)
    assert np.allclose(e, 1.0)


def test_e_meridian_and_western_taper_to_floor():
    lst = 6.0
    # ON the meridian (RA = LST): the ~0.9 shoulder
    e_mer = compute_e_east(np.array([6.0 * 15]), lst, CFG)
    assert e_mer[0] == pytest.approx(CFG.e_meridian)
    # far west (HA >= taper span): floored, never zero
    ha8 = np.array([(lst - 8.0) * 15])          # HA = +8 h
    assert compute_e_east(ha8, lst, CFG)[0] == pytest.approx(CFG.e_floor)
    # monotone non-increasing across the western taper
    has = np.array([0.0, 1.5, 3.0, 4.5, 6.0])
    e_seq = compute_e_east((lst - has) * 15, lst, CFG)
    assert np.all(np.diff(e_seq) < 0) or np.all(np.diff(e_seq) <= 0)
    assert e_seq[-1] == pytest.approx(CFG.e_floor)


def test_e_wraps_hour_angle():
    # LST 2 h, RA 20 h -> naive HA = -18 h, wrapped = +6 h -> deep west
    e = compute_e_east(np.array([20.0 * 15]), 2.0, CFG)
    assert e[0] == pytest.approx(CFG.e_floor, abs=1e-9)
    # LST 22 h, RA 2 h -> naive HA = +20 h, wrapped = -4 h -> east, full weight
    e2 = compute_e_east(np.array([2.0 * 15]), 22.0, CFG)
    assert e2[0] == 1.0


def test_e_unknown_coords_or_no_lst_neutral():
    assert compute_e_east(np.array([np.nan]), 6.0, CFG)[0] == 1.0
    assert np.allclose(compute_e_east(np.array([10.0, 200.0]), None, CFG), 1.0)


def test_e_lst_ha_sanity_via_twilight():
    # LST at evening twilight from LCO is finite and in [0, 24); a target AT
    # that RA (= LST * 15 deg) sits on the meridian -> the e_meridian shoulder
    lst = evening_twilight_lst_hours("2026-08-18")
    assert lst is not None and 0.0 <= lst < 24.0
    e = compute_e_east(np.array([lst * 15.0]), lst, CFG)
    assert e[0] == pytest.approx(CFG.e_meridian)


def test_score_breakdown_includes_e():
    kw = dict(w_prob=np.array([0.9]), w_iaspec=np.array([1.0]),
              salt_c_err=np.array([np.nan]), tns_type=[None],
              z_source=["none"], delta_t=np.array([2.0]),
              z=np.array([0.05]), config=CFG)
    west = compute_score_breakdown(ra=np.array([(6.0 - 8.0) * 15]),
                                   lst_hours=6.0, **kw)
    east = compute_score_breakdown(ra=np.array([(6.0 + 4.0) * 15]),
                                   lst_hours=6.0, **kw)
    none = compute_score_breakdown(**kw)                 # no coords -> neutral
    assert west['e_east'][0] == pytest.approx(CFG.e_floor)
    assert east['e_east'][0] == 1.0 == none['e_east'][0]
    assert west['score'][0] == pytest.approx(east['score'][0] * CFG.e_floor)


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
