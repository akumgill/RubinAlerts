"""S/N-based exposure-time estimator (ETC) for LLAMAS SN spectroscopy.

Digitized from Chris Stubbs's calibration curve "LLAMAS: exposure time to reach
SNR=5 on SN Ia peak vs redshift" (SNR=5 PER PIXEL over 5150-6250 A; 10-min
reference exposure), indexed by SN Ia peak apparent r magnitude. The curve's
solid segment (r<21) is within the LLAMAS fit range; r>21 is extrapolated and
flagged as such.

Two facts from Chris fold in on top of the raw curve:

  * BINNING. SNe have only broad spectral features, and LLAMAS's R~2000 is ~10x
    finer than needed, so we bin ~n_bin pixels in the wavelength direction.
    Photon-limited SNR ~ sqrt(N), so binning n_bin pixels gains sqrt(n_bin) in
    SNR -> to reach the SAME SNR target the net exposure is n_bin x SHORTER
    than the per-pixel curve (n_bin~10 -> ~10x shorter).
  * SNR ~ sqrt(t). A target other than SNR=5 scales the time by (target/5)^2.

    t_net(r) = t_curve_pp5(r) * (target_snr / 5)^2 / n_bin

Break the net exposure into 300-600 s sub-exposures for cosmic-ray rejection
(``split_exposure``). Caveats carried deliberately: the curve is a point-source
SN Ia *peak* calc (no explicit host-galaxy background term, so faint SNe on
bright hosts run optimistic), and r>21 is extrapolated beyond the calibration.

The digitized points are a pixel-extraction of the published PNG (see the _R/_T
note below), good to a few percent within the fit range; the r>21 extrapolated
segment inherits the curve author's extrapolation and is flagged on return.
"""
import bisect
import math

# (SN Ia peak apparent r, minutes to reach SNR=5 PER PIXEL) — PIXEL-DIGITIZED
# from docs/design/figures/llamas_snia_exptime_vs_z.png (blue curve extracted;
# y calibrated on the 10-min/60-min dotted reference lines, x->r via the top
# axis ticks). Cross-checks that held: curve crosses 10 min at r=21.4 and 60 min
# at r=23.2, box bottom reads t=0.0096 (=10^-2). r < 21 is the LLAMAS fit range
# (solid); r >= 21 is extrapolated (dashed).
_R = [17.4, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0,
      21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 24.8]
_T = [0.061, 0.302, 0.478, 0.751, 1.374, 2.315, 3.521, 6.311,
      11.787, 17.998, 28.753, 51.325, 88.291, 146.969, 244.645, 420.846]
_LOGT = [math.log10(t) for t in _T]

FIT_RANGE_MAX_R = 21.0          # solid/dashed boundary on the curve
REFERENCE_SNR = 5.0             # the curve's PER-PIXEL SNR target (Chris's chart)
# target_snr here is the BINNED (post-binning) SNR the observer wants. Chris's
# note gives the per-pixel reference (5) and the binning gain (sqrt(n_bin)); it
# does NOT state the target binned SNR. 10 is a comfortable typing default (t is
# 0.4x the curve time at n_bin=10). OPEN QUESTION for Chris — see notebook.
DEFAULT_TARGET_SNR = 10.0
DEFAULT_N_BIN = 10              # spectral binning (R~2000 -> ~10x too fine)
MIN_EXPOSURE_MINUTES = 10.0     # operational floor. A LLAMAS target switch costs
                                # ~3-6 min modelled (overhead 1 + acquisition 2 +
                                # slew) and ~5 min in practice; a 10-min minimum
                                # keeps the exposure ~2x the switch overhead
                                # (~70% science efficiency) instead of churning
                                # through overhead-dominated ~5-min visits.
MAX_EXPOSURE_MIN = 240.0        # cap (4 h); beyond this a target isn't feasible
DEFAULT_SUB_EXPOSURE_SEC = 300  # cosmic-ray split; 300 or 600 s per Chris


def snr_exposure_minutes(r_mag, target_snr=DEFAULT_TARGET_SNR,
                         n_bin=DEFAULT_N_BIN, max_minutes=MAX_EXPOSURE_MIN):
    """Net LLAMAS exposure (minutes) to reach ``target_snr`` on a binned
    resolution element for a source of apparent ``r_mag``.

    Returns ``(minutes, extrapolated)`` where ``extrapolated`` is True when
    r_mag is outside the LLAMAS fit range (r>21, or brighter than the curve's
    left edge). Returns ``(nan, False)`` for a non-finite magnitude.

    Uses log-linear interpolation of the per-pixel-SNR=5 curve (clamped at the
    endpoints), then applies the (target/5)^2 photon scaling and the /n_bin
    spectral-binning gain.
    """
    if r_mag is None or not math.isfinite(r_mag):
        return float("nan"), False
    r = min(max(r_mag, _R[0]), _R[-1])
    i = min(max(bisect.bisect_right(_R, r) - 1, 0), len(_R) - 2)
    frac = (r - _R[i]) / (_R[i + 1] - _R[i])
    t_pp5 = 10.0 ** (_LOGT[i] + frac * (_LOGT[i + 1] - _LOGT[i]))
    t = t_pp5 * (target_snr / REFERENCE_SNR) ** 2 / max(n_bin, 1)
    t = min(t, max_minutes)
    extrapolated = (r_mag > FIT_RANGE_MAX_R) or (r_mag < _R[0])
    return t, extrapolated


def split_exposure(net_minutes, sub_exposure_sec=DEFAULT_SUB_EXPOSURE_SEC):
    """Break a net exposure into equal sub-exposures for cosmic-ray rejection.

    Returns ``(n_sub, sub_exposure_sec)`` — at least one sub-exposure for any
    positive net time; ``(0, sub_exposure_sec)`` for a non-positive/non-finite
    net time.
    """
    if net_minutes is None or not math.isfinite(net_minutes) or net_minutes <= 0:
        return 0, sub_exposure_sec
    n = max(1, math.ceil(net_minutes * 60.0 / sub_exposure_sec))
    return n, sub_exposure_sec
