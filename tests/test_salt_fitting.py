"""Tests for the SALT2 typing/phase integration.

Covers:
  1. Pure logic (no sncosmo): salt_z_policy, choose_best_fit, to_sncosmo_band,
     resolve_salt_mode.
  2. run_tonight wiring with a monkeypatched fit_salt (no network, no sncosmo
     model needed).
  3. Merit/summary-table plumbing (w_salt, absolute_mag anti-circularity,
     new salt_* columns).
  4. Guarded integration tests against the real salt2-extended model — these
     SKIP when the sncosmo cache is cold (model would need a download) and
     run with sockets blocked so they can never hit the network.
"""

import socket

import numpy as np
import pandas as pd
import pytest

from core.peak_fitting import (
    AB_ZP_NJY,
    HAS_SNCOSMO,
    choose_best_fit,
    fit_salt,
    load_salt_model,
    salt_z_policy,
    to_sncosmo_band,
)
import run_tonight as rt


# ---------------------------------------------------------------------------
# Fake fit-result builders (shared by the pure-logic and wiring tests)
# ---------------------------------------------------------------------------

MJD_MIN, MJD_MAX = 61000.0, 61019.0


def _vil(ok=True, peak_mjd=61012.0, peak_mag=20.0, n_bands_fit=2,
         shared_t0=61000.0):
    if not ok:
        return {'per_band': {}, 'best': None, 'method': 'villar_multiband',
                'shared_t0': np.nan, 'n_bands_fit': 0}
    best = {'band': 'r', 'status': 'ok', 'peak_mjd': peak_mjd,
            'peak_mag': peak_mag, 'peak_flux': 1e4, 'chi2_dof': 1.5,
            'method': 'villar_multiband'}
    return {'per_band': {'r': best}, 'best': best,
            'method': 'villar_multiband', 'shared_t0': shared_t0,
            'n_bands_fit': n_bands_fit}


def _par(ok=True, bands_ok=2, peak_mjd=61011.0, peak_mag=20.2):
    if not ok:
        return {'per_band': {}, 'best': None, 'method': 'parabola'}
    best = {'band': 'r', 'status': 'ok', 'peak_mjd': peak_mjd,
            'peak_mag': peak_mag, 'chi2_dof': 2.0}
    per_band = {}
    for i, b in enumerate(['r', 'g', 'i', 'z']):
        if i < bands_ok:
            per_band[b] = {'band': b, 'status': 'ok'}
        else:
            per_band[b] = {'band': b, 'status': 'fit_failed'}
    return {'per_band': per_band, 'best': best, 'method': 'parabola'}


def _salt(status='ok', chi2_dof=1.0, t0_err=0.5, t0=61015.0,
          peak_mag_obs=19.3, z=0.08, z_railed=False, peak_mag_B=19.5):
    return {'status': status, 'method': 'salt', 'model': 'salt2-extended',
            't0': t0, 't0_err': t0_err, 'x0': 1e-4, 'x1': 0.5,
            'x1_err': 0.3, 'c': 0.02, 'c_err': 0.04, 'z': z,
            'z_railed': z_railed, 'peak_mag_obs': peak_mag_obs,
            'peak_band_obs': 'r', 'peak_mag_B': peak_mag_B,
            'chi2': chi2_dof * 20, 'ndof': 20, 'chi2_dof': chi2_dof,
            'mwebv': 0.02, 'n_points': 20, 'n_bands': 2}


# ---------------------------------------------------------------------------
# 1. Pure logic: salt_z_policy
# ---------------------------------------------------------------------------

class TestSaltZPolicy:
    def test_tns_specz_is_fixed(self):
        z_fixed, z_bounds = salt_z_policy(0.08, 'tns_specz')
        assert z_fixed == pytest.approx(0.08)
        assert z_bounds is None

    def test_legacy_photz_gives_two_sigma_bounds(self):
        z_fixed, z_bounds = salt_z_policy(0.2, 'legacy_photz')
        assert z_fixed is None
        sig = max(0.03, 0.05 * 1.2)
        assert z_bounds[0] == pytest.approx(0.2 - 2 * sig)
        assert z_bounds[1] == pytest.approx(0.2 + 2 * sig)

    def test_legacy_photz_lower_bound_clamped(self):
        # Small photo-z: lower bound must not go below 0.005
        _, z_bounds = salt_z_policy(0.02, 'legacy_photz')
        assert z_bounds[0] == pytest.approx(0.005)

    def test_no_source_gives_bright_target_box(self):
        for src in (None, '', 'none'):
            z_fixed, z_bounds = salt_z_policy(np.nan, src)
            assert z_fixed is None
            assert z_bounds == [0.005, 0.15]

    def test_specz_source_with_nan_z_falls_back_to_box(self):
        z_fixed, z_bounds = salt_z_policy(np.nan, 'tns_specz')
        assert z_fixed is None
        assert z_bounds == [0.005, 0.15]


# ---------------------------------------------------------------------------
# 1. Pure logic: choose_best_fit decision matrix
# ---------------------------------------------------------------------------

class TestChooseBestFit:
    def test_salt_wins_when_good(self):
        best, method = choose_best_fit(_vil(), _par(), _salt(chi2_dof=1.0),
                                       2, MJD_MIN, MJD_MAX)
        assert method == 'salt'
        assert best['peak_mjd'] == pytest.approx(61015.0)
        assert best['peak_mag'] == pytest.approx(19.3)
        assert best['band'] == 'r'
        assert best['status'] == 'ok'

    def test_villar_wins_on_bad_salt_chi2(self):
        best, method = choose_best_fit(_vil(), _par(), _salt(chi2_dof=5.0),
                                       2, MJD_MIN, MJD_MAX)
        assert method == 'villar_mb'
        assert best['peak_mjd'] == pytest.approx(61012.0)

    def test_villar_wins_on_bad_salt_t0_err(self):
        best, method = choose_best_fit(_vil(), _par(), _salt(t0_err=8.0),
                                       2, MJD_MIN, MJD_MAX)
        assert method == 'villar_mb'

    def test_villar_wins_on_extrapolated_salt_t0(self):
        # t0 more than 10 d before the first point → not anchored by data
        _, method = choose_best_fit(_vil(), _par(),
                                    _salt(t0=MJD_MIN - 30.0),
                                    2, MJD_MIN, MJD_MAX)
        assert method == 'villar_mb'
        # ... and more than 20 d after the last point
        _, method = choose_best_fit(_vil(), _par(),
                                    _salt(t0=MJD_MAX + 30.0),
                                    2, MJD_MIN, MJD_MAX)
        assert method == 'villar_mb'

    def test_salt_rescues_when_villar_and_parabola_fail(self):
        best, method = choose_best_fit(_vil(ok=False), _par(ok=False),
                                       _salt(), 2, MJD_MIN, MJD_MAX)
        assert method == 'salt'
        assert best['peak_mag'] == pytest.approx(19.3)

    def test_all_fail(self):
        best, method = choose_best_fit(_vil(ok=False), _par(ok=False),
                                       _salt(status='fit_failed'),
                                       2, MJD_MIN, MJD_MAX)
        assert best is None
        assert method == 'none'

    def test_no_salt_falls_through_to_existing_preference(self):
        # Villar preferred when it passes the band gate...
        _, method = choose_best_fit(_vil(), _par(), None, 2,
                                    MJD_MIN, MJD_MAX)
        assert method == 'villar_mb'
        # ...parabola when Villar has too few bands...
        best, method = choose_best_fit(_vil(n_bands_fit=1), _par(bands_ok=2),
                                       None, 2, MJD_MIN, MJD_MAX)
        assert method == 'parabola'
        assert best['peak_mjd'] == pytest.approx(61011.0)
        # ...nothing when parabola also has too few bands.
        best, method = choose_best_fit(_vil(n_bands_fit=1), _par(bands_ok=1),
                                       None, 2, MJD_MIN, MJD_MAX)
        assert method == 'none'
        assert best is None

    def test_salt_without_finite_peak_mag_falls_through(self):
        _, method = choose_best_fit(_vil(), _par(),
                                    _salt(peak_mag_obs=np.nan),
                                    2, MJD_MIN, MJD_MAX)
        assert method == 'villar_mb'


# ---------------------------------------------------------------------------
# 1. Pure logic: band mapping
# ---------------------------------------------------------------------------

class TestToSncosmoBand:
    @pytest.mark.parametrize('band,survey,expected', [
        ('u', 'Rubin', 'lsstu'),
        ('g', 'Rubin', 'lsstg'),
        ('r', 'Rubin', 'lsstr'),
        ('i', 'Rubin', 'lssti'),
        ('z', 'Rubin', 'lsstz'),
        ('y', 'Rubin', 'lssty'),
        ('g', 'ZTF', 'ztfg'),
        ('r', 'ZTF', 'ztfr'),
        ('i', 'ZTF', 'ztfi'),
        ('ATLAS-c', 'ATLAS', 'atlasc'),
        ('ATLAS-o', 'ATLAS', 'atlaso'),
    ])
    def test_known_mappings(self, band, survey, expected):
        assert to_sncosmo_band(band, survey) == expected

    def test_unknowns_are_dropped(self):
        assert to_sncosmo_band('w', 'ATLAS') is None       # unknown band
        assert to_sncosmo_band('r', 'Gaia') is None        # unknown survey
        assert to_sncosmo_band(None, 'Rubin') is None
        assert to_sncosmo_band(np.nan, 'Rubin') is None

    def test_missing_survey_defaults_to_rubin(self):
        assert to_sncosmo_band('r', None) == 'lsstr'
        assert to_sncosmo_band('r', '') == 'lsstr'


# ---------------------------------------------------------------------------
# 1. Pure logic: SALT default policy (ddf off, wide on, --no-salt wins)
# ---------------------------------------------------------------------------

class TestResolveSaltMode:
    def test_ddf_default_off(self):
        assert rt.resolve_salt_mode('ddf', False, False, True) is False

    def test_wide_default_on(self):
        assert rt.resolve_salt_mode('wide', False, False, True) is True

    def test_use_salt_opts_in_for_ddf(self):
        assert rt.resolve_salt_mode('ddf', True, False, True) is True

    def test_no_salt_wins_everywhere(self):
        assert rt.resolve_salt_mode('wide', False, True, True) is False
        assert rt.resolve_salt_mode('ddf', True, True, True) is False

    def test_requires_sncosmo(self):
        assert rt.resolve_salt_mode('wide', True, False, False) is False


# ---------------------------------------------------------------------------
# 2. run_tonight wiring (monkeypatched fit_salt — no sncosmo model, no network)
# ---------------------------------------------------------------------------

def _make_lc(mjd0=61000.0, n_nights=20):
    """SN-like 2-band Rubin light curve that passes all quality cuts."""
    rows = []
    mjd = mjd0 + np.arange(n_nights, dtype=float)
    flux = 5000.0 + 3000.0 * np.sin(np.linspace(0, np.pi, n_nights))
    for b in ('g', 'r'):
        for t, f in zip(mjd, flux):
            rows.append({'mjd': t, 'flux': f, 'flux_err': 100.0,
                         'band': b, 'survey': 'Rubin',
                         'source': 'detection'})
    return pd.DataFrame(rows)


class _FakeFink:
    """Stands in for FinkLSSTClient in fetch_and_fit (no network)."""

    def __init__(self, lc_by_did):
        self.lc_by_did = lc_by_did

    def get_light_curve(self, did, include_forced=True):
        lc = self.lc_by_did.get(str(did))
        return lc.copy() if lc is not None else pd.DataFrame()


def _candidates(dids):
    return pd.DataFrame([
        {'diaObjectId': did, 'ra': 150.0 + i, 'dec': -30.0}
        for i, did in enumerate(dids)
    ])


@pytest.fixture
def salt_calls(monkeypatch):
    """Patch run_tonight.fit_salt with a fake; records call kwargs."""
    calls = []

    def fake_fit_salt(lc_df, model_name='salt2-extended', z=None,
                      z_bounds=None, mwebv=None, clean=True):
        calls.append({'z': z, 'z_bounds': z_bounds, 'mwebv': mwebv,
                      'model_name': model_name})
        return _salt()

    monkeypatch.setattr(rt, 'fit_salt', fake_fit_salt)
    monkeypatch.setattr(rt, 'HAS_SNCOSMO', True)
    return calls


def _run_fetch_and_fit(monkeypatch, dids=('obj1',), vil=None, par=None,
                       mjd_now=61020.0, **kwargs):
    monkeypatch.setattr(rt, 'fit_villar_multiband',
                        lambda lc: vil if vil is not None else _vil())
    monkeypatch.setattr(rt, 'fit_parabola',
                        lambda lc: par if par is not None else _par())
    lc_by_did = {str(d): _make_lc() for d in dids}
    fink = _FakeFink(lc_by_did)
    return rt.fetch_and_fit(fink, _candidates(list(dids)), mjd_now,
                            fetch_ztf=False, fetch_atlas=False, **kwargs)


class TestFetchAndFitSaltWiring:
    def test_salt_wins_and_supplies_peak(self, monkeypatch, salt_calls):
        results = _run_fetch_and_fit(
            monkeypatch, use_salt=True,
            redshifts={'obj1': {'z': 0.08, 'source': 'tns_specz'}},
            ebv_lookup={'obj1': 0.02})
        assert 'obj1' in results
        r = results['obj1']
        assert r['fit_method'] == 'salt'
        assert r['peak_mjd'] == pytest.approx(61015.0)   # salt t0
        assert r['peak_mag'] == pytest.approx(19.3)      # salt peak_mag_obs
        assert r['peak_band'] == 'r'
        assert r['salt_rescued'] is False                # generic gate passed
        assert r['salt']['t0_err'] == pytest.approx(0.5)
        assert r['salt']['z_railed'] is False
        # z policy: tns_specz → fixed z, no bounds; mwebv forwarded
        assert salt_calls == [{'z': 0.08, 'z_bounds': None, 'mwebv': 0.02,
                               'model_name': 'salt2-extended'}]

    def test_photz_source_passes_bounds_not_fixed_z(self, monkeypatch,
                                                    salt_calls):
        _run_fetch_and_fit(
            monkeypatch, use_salt=True,
            redshifts={'obj1': {'z': 0.2, 'source': 'legacy_photz'}})
        assert len(salt_calls) == 1
        assert salt_calls[0]['z'] is None
        assert salt_calls[0]['z_bounds'] == pytest.approx([0.08, 0.32])

    def test_old_style_float_redshift_still_fixed(self, monkeypatch,
                                                  salt_calls):
        _run_fetch_and_fit(monkeypatch, use_salt=True,
                           redshifts={'obj1': 0.08})
        assert salt_calls[0]['z'] == pytest.approx(0.08)
        assert salt_calls[0]['z_bounds'] is None

    def test_rise_time_gate_skipped_for_salt(self, monkeypatch):
        # SALT t0 3 d after the first detection → rise-time proxy 3 d < 5 d
        # minimum, which would kill a villar/parabola fit; SALT is exempt.
        monkeypatch.setattr(rt, 'HAS_SNCOSMO', True)
        monkeypatch.setattr(
            rt, 'fit_salt',
            lambda *a, **k: _salt(t0=61003.0, peak_mag_obs=19.5))
        monkeypatch.setattr(rt, 'fit_villar_multiband', lambda lc: _vil(ok=False))
        monkeypatch.setattr(rt, 'fit_parabola', lambda lc: _par(ok=False))
        fink = _FakeFink({'obj1': _make_lc()})
        results = rt.fetch_and_fit(fink, _candidates(['obj1']), 61020.0,
                                   fetch_ztf=False, fetch_atlas=False,
                                   use_salt=True)
        assert 'obj1' in results
        assert results['obj1']['fit_method'] == 'salt'

    def test_rise_time_gate_still_applies_to_villar(self, monkeypatch):
        # Same 3 d rise through the Villar path → object is dropped.
        vil = _vil(peak_mjd=61003.0, shared_t0=61000.0)
        results = _run_fetch_and_fit(monkeypatch, vil=vil, par=_par(ok=False),
                                     use_salt=False)
        assert results == {}

    def test_max_phase_gate_still_applies_to_salt(self, monkeypatch):
        # SALT peak 40 d before mjd_now > max_phase_days=25 → dropped even
        # though the SALT fit itself is good.
        monkeypatch.setattr(rt, 'HAS_SNCOSMO', True)
        monkeypatch.setattr(rt, 'fit_salt',
                            lambda *a, **k: _salt(t0=61005.0))
        monkeypatch.setattr(rt, 'fit_villar_multiband', lambda lc: _vil(ok=False))
        monkeypatch.setattr(rt, 'fit_parabola', lambda lc: _par(ok=False))
        fink = _FakeFink({'obj1': _make_lc()})
        results = rt.fetch_and_fit(fink, _candidates(['obj1']), 61045.0,
                                   fetch_ztf=False, fetch_atlas=False,
                                   use_salt=True)
        assert results == {}

    def test_salt_rescued_flag_set(self, monkeypatch, salt_calls):
        results = _run_fetch_and_fit(monkeypatch, vil=_vil(ok=False),
                                     par=_par(ok=False), use_salt=True)
        assert results['obj1']['fit_method'] == 'salt'
        assert results['obj1']['salt_rescued'] is True

    def test_rescue_cap_honored(self, monkeypatch, salt_calls):
        results = _run_fetch_and_fit(monkeypatch, dids=('obj1', 'obj2'),
                                     vil=_vil(ok=False), par=_par(ok=False),
                                     use_salt=True, salt_rescue_cap=1)
        # Only the first (encounter order) got a rescue attempt
        assert len(salt_calls) == 1
        assert list(results.keys()) == ['obj1']

    def test_tier1_salt_not_counted_against_rescue_cap(self, monkeypatch,
                                                       salt_calls):
        # Object passes the generic gate → SALT runs even with cap 0.
        results = _run_fetch_and_fit(monkeypatch, use_salt=True,
                                     salt_rescue_cap=0)
        assert len(salt_calls) == 1
        assert results['obj1']['fit_method'] == 'salt'
        assert results['obj1']['salt_rescued'] is False

    def test_use_salt_false_never_calls_fit_salt(self, monkeypatch,
                                                 salt_calls):
        results = _run_fetch_and_fit(monkeypatch, use_salt=False)
        assert salt_calls == []
        assert results['obj1']['fit_method'] == 'villar_mb'
        assert results['obj1']['salt'] is None


# ---------------------------------------------------------------------------
# 3. Merit / summary-table plumbing
# ---------------------------------------------------------------------------

def _fit_entry(salt=None, salt_rescued=False, peak_mag=20.5, delta_t=5.0):
    return {
        'diaObjectId': 'obj1',
        'peak_mag': peak_mag, 'peak_mjd': 61015.0, 'peak_band': 'r',
        'delta_t': delta_t, 'rise_time': 15.0,
        'fit_method': 'salt' if salt else 'villar_mb',
        'salt_rescued': salt_rescued,
        'n_points': 20, 'n_bands': 2, 'surveys': ['Rubin'],
        'n_ztf': 0, 'n_atlas': 0, 'salt': salt,
        'parabola': {'per_band': {}, 'best': None},
        'villar': {'per_band': {}, 'best': None},
    }


def _summary_candidates(z_source='tns_specz'):
    return pd.DataFrame([{
        'diaObjectId': 'obj1', 'ra': 150.0, 'dec': -30.0,
        'sn_score': 0.9, 'num_brokers': 1, 'z_source': z_source,
    }])


class TestSummaryTableSalt:
    def test_w_salt_bonus_for_good_chi2(self):
        summary = rt.build_summary_table(
            _summary_candidates(), {'obj1': _fit_entry(salt=_salt(chi2_dof=1.0))},
            61020.0)
        row = summary.iloc[0]
        assert row['w_salt'] == pytest.approx(1.2, abs=0.01)

    def test_w_salt_penalty_for_bad_chi2(self):
        summary = rt.build_summary_table(
            _summary_candidates(), {'obj1': _fit_entry(salt=_salt(chi2_dof=6.0))},
            61020.0)
        assert summary.iloc[0]['w_salt'] <= 0.6

    def test_new_salt_columns_present(self):
        salt = _salt(t0=61015.0, t0_err=0.5, z_railed=True)
        summary = rt.build_summary_table(
            _summary_candidates(),
            {'obj1': _fit_entry(salt=salt, salt_rescued=True)},
            61020.0)
        row = summary.iloc[0]
        assert row['salt_t0'] == pytest.approx(61015.0)
        assert row['salt_t0_err'] == pytest.approx(0.5)
        assert bool(row['salt_z_railed']) is True
        assert bool(row['salt_rescued']) is True

    def test_absolute_mag_from_salt_with_external_z(self):
        redshifts = {'obj1': {'redshift': 0.08, 'distmod': 37.67,
                              'ned_name': 'payload:tns_specz',
                              'separation_arcsec': np.nan}}
        salt = _salt(peak_mag_B=18.9)
        summary = rt.build_summary_table(
            _summary_candidates(z_source='tns_specz'),
            {'obj1': _fit_entry(salt=salt)}, 61020.0, redshifts=redshifts)
        assert summary.iloc[0]['absolute_mag'] == pytest.approx(18.9 - 37.67)

    def test_absolute_mag_from_salt_with_photz(self):
        redshifts = {'obj1': {'redshift': 0.1, 'distmod': 38.2,
                              'ned_name': 'payload:legacy_photz',
                              'separation_arcsec': np.nan}}
        summary = rt.build_summary_table(
            _summary_candidates(z_source='legacy_photz'),
            {'obj1': _fit_entry(salt=_salt(peak_mag_B=18.9))},
            61020.0, redshifts=redshifts)
        assert summary.iloc[0]['absolute_mag'] == pytest.approx(18.9 - 38.2)

    def test_absolute_mag_anticircular_when_salt_floated_z(self):
        # No external z provenance → the SALT M_B is circular; fall back to
        # the observed peak mag minus distmod.
        redshifts = {'obj1': {'redshift': 0.08, 'distmod': 37.67,
                              'ned_name': 'x', 'separation_arcsec': 1.0}}
        summary = rt.build_summary_table(
            _summary_candidates(z_source='none'),
            {'obj1': _fit_entry(salt=_salt(peak_mag_B=18.9), peak_mag=20.5)},
            61020.0, redshifts=redshifts)
        assert summary.iloc[0]['absolute_mag'] == pytest.approx(20.5 - 37.67)

    def test_absolute_mag_fallback_without_salt(self):
        redshifts = {'obj1': {'redshift': 0.08, 'distmod': 37.67,
                              'ned_name': 'x', 'separation_arcsec': 1.0}}
        summary = rt.build_summary_table(
            _summary_candidates(z_source='tns_specz'),
            {'obj1': _fit_entry(salt=None, peak_mag=20.5)},
            61020.0, redshifts=redshifts)
        assert summary.iloc[0]['absolute_mag'] == pytest.approx(20.5 - 37.67)


# ---------------------------------------------------------------------------
# 4. Guarded integration tests against the real salt2-extended model.
#    Sockets are blocked for the whole fixture: if the sncosmo cache is cold
#    the model load fails and we SKIP — a download can never happen here.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def salt_model_or_skip():
    if not HAS_SNCOSMO:
        pytest.skip('sncosmo not installed')
    import sncosmo

    real_connect = socket.socket.connect

    def _blocked(self, *args, **kwargs):
        raise OSError('network disabled in tests')

    socket.socket.connect = _blocked
    try:
        model = load_salt_model()
        if model is None:
            pytest.skip('salt2-extended model cache is cold '
                        '(would require a download)')
        try:
            for b in ('lsstg', 'lsstr', 'bessellb'):
                sncosmo.get_bandpass(b)
        except Exception:
            pytest.skip('sncosmo bandpass cache is cold '
                        '(would require a download)')
    finally:
        socket.socket.connect = real_connect
    return model


@pytest.fixture
def no_network(monkeypatch):
    def _blocked(self, *args, **kwargs):
        raise OSError('network disabled in tests')
    monkeypatch.setattr(socket.socket, 'connect', _blocked)


def _synthesize_ia_lc(model, z=0.08, t0=61000.0, x1=0.5, c=0.0, seed=42):
    """3-day-cadence 2-band Rubin LC drawn from the SALT model itself."""
    import copy as _copy
    syn = _copy.copy(model)
    syn.set(z=z, t0=t0, x0=1e-4, x1=x1, c=c, mwebv=0.0)
    rng = np.random.default_rng(seed)
    rows = []
    for band, sb in (('g', 'lsstg'), ('r', 'lsstr')):
        mjd = np.arange(t0 - 12.0, t0 + 30.0, 3.0)
        flux = syn.bandflux(sb, mjd, zp=AB_ZP_NJY, zpsys='ab')
        err = np.maximum(0.05 * np.abs(flux), 50.0)
        flux = flux + rng.normal(0, err)
        for t, f, e in zip(mjd, flux, err):
            rows.append({'mjd': t, 'flux': f, 'flux_err': e,
                         'band': band, 'survey': 'Rubin'})
    return pd.DataFrame(rows)


class TestSaltIntegration:
    T0_TRUE = 61000.0

    def test_recovers_t0_with_fixed_z(self, salt_model_or_skip, no_network):
        lc = _synthesize_ia_lc(salt_model_or_skip, z=0.08, t0=self.T0_TRUE)
        res = fit_salt(lc, z=0.08)
        assert res['status'] == 'ok'
        assert abs(res['t0'] - self.T0_TRUE) < 1.5
        assert np.isfinite(res['t0_err'])
        assert np.isfinite(res['peak_mag_obs'])
        assert res['peak_band_obs'] in ('g', 'r')
        assert np.isfinite(res['peak_mag_B'])
        assert res['z_railed'] is False
        # And the decision layer picks it up
        best, method = choose_best_fit(None, None, res, 2,
                                       lc['mjd'].min(), lc['mjd'].max())
        assert method == 'salt'
        assert best['peak_mjd'] == pytest.approx(res['t0'])

    def test_non_ia_rejected_by_chi2(self, salt_model_or_skip, no_network):
        # Slow linear riser over 60 d — nothing like an Ia
        rng = np.random.default_rng(3)
        rows = []
        mjd = np.arange(60960.0, 61021.0, 3.0)
        for band in ('g', 'r'):
            flux = 500.0 + (mjd - mjd.min()) * 500.0
            err = np.maximum(0.05 * flux, 50.0)
            f = flux + rng.normal(0, err)
            for t, fv, e in zip(mjd, f, err):
                rows.append({'mjd': t, 'flux': fv, 'flux_err': e,
                             'band': band, 'survey': 'Rubin'})
        lc = pd.DataFrame(rows)
        res = fit_salt(lc, z=None, z_bounds=[0.005, 0.15])
        assert res['status'] != 'ok' or res['chi2_dof'] > 3.0
        _, method = choose_best_fit(None, None, res, 2,
                                    lc['mjd'].min(), lc['mjd'].max())
        assert method == 'none'

    def test_unknown_bands_yield_insufficient_data(self, no_network):
        # All points from an unmapped survey → dropped before any model
        # load, so this works even with a cold cache and no network.
        if not HAS_SNCOSMO:
            pytest.skip('sncosmo not installed')
        lc = pd.DataFrame({
            'mjd': np.arange(61000.0, 61020.0, 2.0),
            'flux': np.full(10, 5000.0),
            'flux_err': np.full(10, 100.0),
            'band': ['V'] * 10,
            'survey': ['Gaia'] * 10,
        })
        res = fit_salt(lc)
        assert res['status'] == 'insufficient_data'
