"""Tests for the S/N-based exposure estimator (Chris's LLAMAS SN Ia curve)."""
import math

import pytest

from core import snr_etc


def test_binning_shortens_by_n_bin():
    # SNR=5 target, no binning, at a curve knot (r=23.4 -> 80 min per pixel).
    t_raw, _ = snr_etc.snr_exposure_minutes(23.4, target_snr=5, n_bin=1)
    assert t_raw == pytest.approx(80.0, rel=0.02)
    # n_bin=10 -> 10x shorter (Chris: bin ~10 px, gain sqrt(10) in SNR).
    t_bin, _ = snr_etc.snr_exposure_minutes(23.4, target_snr=5, n_bin=10)
    assert t_bin == pytest.approx(8.0, rel=0.02)


def test_target_snr_scales_as_square():
    # SNR ~ sqrt(t): doubling the SNR target quadruples the time.
    t5, _ = snr_etc.snr_exposure_minutes(20.0, target_snr=5, n_bin=1)
    t10, _ = snr_etc.snr_exposure_minutes(20.0, target_snr=10, n_bin=1)
    assert t10 == pytest.approx(4.0 * t5, rel=1e-6)


def test_extrapolated_flag_beyond_fit_range():
    _, ex_faint = snr_etc.snr_exposure_minutes(23.0)   # r > 21
    _, ex_ok = snr_etc.snr_exposure_minutes(20.0)      # r < 21
    assert ex_faint is True
    assert ex_ok is False


def test_log_linear_interpolation_between_knots():
    # midpoint in mag between two knots should sit between their times, and
    # closer (in log) to the geometric mean than the arithmetic mean.
    t_lo, _ = snr_etc.snr_exposure_minutes(22.0, n_bin=1)   # 15 min knot
    t_hi, _ = snr_etc.snr_exposure_minutes(22.5, n_bin=1)   # 25 min knot
    t_mid, _ = snr_etc.snr_exposure_minutes(22.25, n_bin=1)
    assert t_lo < t_mid < t_hi
    assert t_mid == pytest.approx(math.sqrt(15.0 * 25.0), rel=0.02)


def test_faint_z08_is_feasible_with_binning():
    # z~0.8 is r~23.4; with binning it should be a short, feasible net exposure.
    t, extrap = snr_etc.snr_exposure_minutes(23.4)   # defaults: SNR=5, n_bin=10
    assert 5.0 < t < 15.0
    assert extrap is True


def test_max_exposure_cap_and_nan():
    t, _ = snr_etc.snr_exposure_minutes(24.8, target_snr=20, n_bin=1)
    assert t == snr_etc.MAX_EXPOSURE_MIN            # capped
    assert math.isnan(snr_etc.snr_exposure_minutes(float("nan"))[0])


def test_split_exposure_for_cosmic_rays():
    # 8-min net at 300 s -> 2 sub-exposures (>=2 good for CR rejection).
    assert snr_etc.split_exposure(8.0, 300) == (2, 300)
    # 18-min net at 300 s -> 4 sub-exposures.
    assert snr_etc.split_exposure(18.0, 300) == (4, 300)
    # 8-min net at 600 s -> 1 sub-exposure (rounds up).
    assert snr_etc.split_exposure(8.0, 600) == (1, 600)
    # non-positive -> zero sub-exposures.
    assert snr_etc.split_exposure(0.0) == (0, snr_etc.DEFAULT_SUB_EXPOSURE_SEC)
