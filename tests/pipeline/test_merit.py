"""Tests for merit correctness fixes R1 (moon penalty) and R6 (broker coverage).

All inputs are in-memory scalars/arrays — no network, DB, or API access.
"""

import numpy as np

from core.magellan_planning import compute_merit_breakdown, _mag_weight
from core.ddf_fields import (
    max_possible_brokers,
    MAX_BROKERS_ZTF_COVERAGE,
    MAX_BROKERS_LSST_ONLY,
    ZTF_MIN_DEC,
)


# Common, fully-observable merit inputs (near peak, optimal magnitude).
COMMON = dict(
    delta_t=0.0,
    peak_mag=20.5,
    ia_prob=0.6,
    host_morphology='unknown',
    extinction_ebv=0.03,
)


# --- W1: moon penalty folds into the ranking merit ------------------------

def test_moon_penalty_scales_merit_linearly():
    """merit(moon=0.5) == merit(moon=1.0) * 0.5 and w_moon reports the penalty."""
    no_moon = compute_merit_breakdown(num_brokers=1, max_possible_brokers=4,
                                      moon_penalty=1.0, **COMMON)
    half_moon = compute_merit_breakdown(num_brokers=1, max_possible_brokers=4,
                                        moon_penalty=0.5, **COMMON)

    assert float(no_moon['w_moon']) == 1.0
    assert float(half_moon['w_moon']) == 0.5
    assert np.isclose(float(half_moon['merit']),
                      float(no_moon['merit']) * 0.5)


# --- W2: coverage-aware broker bonus --------------------------------------

def test_max_possible_brokers_by_declination():
    """Southern (dec <= -32) fields are LSST-only; equatorial get ZTF coverage."""
    assert max_possible_brokers(2.0) == MAX_BROKERS_ZTF_COVERAGE
    assert max_possible_brokers(-47.0) == MAX_BROKERS_LSST_ONLY
    assert max_possible_brokers(ZTF_MIN_DEC) == MAX_BROKERS_LSST_ONLY  # boundary
    # Array form preserves per-element classification.
    arr = max_possible_brokers(np.array([2.0, -47.0]))
    assert list(arr) == [MAX_BROKERS_ZTF_COVERAGE, MAX_BROKERS_LSST_ONLY]


def test_single_broker_southern_and_equatorial_tie():
    """A 1-broker southern field is NOT penalised vs a 1-broker equatorial field."""
    equatorial = compute_merit_breakdown(
        num_brokers=1, max_possible_brokers=max_possible_brokers(2.0), **COMMON)
    southern = compute_merit_breakdown(
        num_brokers=1, max_possible_brokers=max_possible_brokers(-47.0), **COMMON)

    assert np.isclose(float(equatorial['w_broker']), float(southern['w_broker']))
    # Single broker earns no bonus regardless of region.
    assert np.isclose(float(equatorial['w_broker']), 1.0)


def test_multibroker_equatorial_beats_single_broker():
    """A multi-broker equatorial candidate gets a higher w_broker than a single."""
    single = compute_merit_breakdown(
        num_brokers=1, max_possible_brokers=max_possible_brokers(2.0), **COMMON)
    multi = compute_merit_breakdown(
        num_brokers=3, max_possible_brokers=max_possible_brokers(2.0), **COMMON)

    assert float(multi['w_broker']) > float(single['w_broker'])
    # Full coverage caps at the historical +30% bonus.
    full = compute_merit_breakdown(
        num_brokers=MAX_BROKERS_ZTF_COVERAGE,
        max_possible_brokers=MAX_BROKERS_ZTF_COVERAGE, **COMMON)
    assert np.isclose(float(full['w_broker']), 1.3)


# --- W3: soft top-hat magnitude weight (fixes bright-SN penalty) ----------

def test_mag_weight_plateau_is_flat():
    """Full weight across the LLAMAS followable plateau [18, 21]."""
    for m in (18.0, 18.5, 19.0, 20.0, 20.5, 21.0):
        assert np.isclose(float(_mag_weight(m)), 1.0, atol=1e-9)


def test_mag_weight_bright_penalty_is_gone():
    """Bright SNe are NOT penalised: mag 19 ~ 1.0 (was 0.24 under old Gaussian)."""
    assert float(_mag_weight(19.0)) >= 0.9
    assert np.isclose(float(_mag_weight(19.0)), 1.0, atol=1e-9)


def test_mag_weight_faint_rolloff():
    """Faint side rolls off monotonically toward the ~mag 22-23 exposure wall."""
    assert np.isclose(float(_mag_weight(21.5)), 0.7788, atol=1e-3)
    assert np.isclose(float(_mag_weight(22.0)), 0.3679, atol=1e-3)
    assert float(_mag_weight(23.0)) < 0.05
    assert float(_mag_weight(24.0)) < 0.01
    faint = [float(_mag_weight(m)) for m in (21.0, 21.5, 22.0, 23.0, 24.0)]
    assert all(a > b for a, b in zip(faint, faint[1:]))  # monotonic decreasing


def test_mag_weight_gentle_bright_rolloff():
    """Bright-side roll-off is gentle (saturation/low-z), not a steep penalty."""
    assert np.isclose(float(_mag_weight(18.0)), 1.0, atol=1e-9)
    assert 0.6 < float(_mag_weight(17.0)) < 0.9
    assert float(_mag_weight(16.0)) < float(_mag_weight(17.0))


def test_mag_weight_vectorized():
    """Array input returns an array of the same shape with expected values."""
    mags = np.array([18.0, 19.0, 20.5, 21.0, 22.0])
    w = _mag_weight(mags)
    assert isinstance(w, np.ndarray)
    assert w.shape == mags.shape
    np.testing.assert_allclose(w, [1.0, 1.0, 1.0, 1.0, 0.3679], atol=1e-3)


def test_breakdown_uses_soft_tophat_mag_weight():
    """A bright near-peak candidate (mag 19) now scores w_mag ~ 1.0, not ~0.24."""
    common = dict(delta_t=0.0, ia_prob=0.6, host_morphology='unknown',
                  extinction_ebv=0.03)
    bright = compute_merit_breakdown(peak_mag=19.0, **common)
    faint = compute_merit_breakdown(peak_mag=22.0, **common)

    assert np.isclose(float(bright['w_mag']), 1.0, atol=1e-9)
    assert np.isclose(float(faint['w_mag']), 0.3679, atol=1e-3)
    # Brighter near-peak target now outranks a fainter one (was inverted/penalised).
    assert float(bright['merit']) > float(faint['merit'])
