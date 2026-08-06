"""Tests for the multi-type template tournament (SALT2 vs CC templates).

The template fits need the sncosmo sources on disk (salt2-extended +
nugent-*); tests that fit are skipped when a source is unavailable
(offline machine without a warmed SNCOSMO_DATA_DIR cache) so the suite
stays offline-safe.
"""
import numpy as np
import pandas as pd
import pytest

from core import peak_fitting as pf
from core.peak_fitting import (CC_TEMPLATES, fit_template,
                               run_template_tournament, load_salt_model)


def _sources_available(*names):
    if not pf.HAS_SNCOSMO:
        return False
    return all(load_salt_model(n) is not None for n in names)


def _synthesize_lc(model_name, z=0.05, seed=0):
    """Sample a ZTF g+r light curve from an sncosmo source (2% noise)."""
    import copy
    import sncosmo
    rng = np.random.default_rng(seed)
    model = copy.copy(load_salt_model(model_name))
    model.set(z=z, mwebv=0.0, t0=60000.0)
    if 'x0' in model.param_names:
        model.set_source_peakabsmag(-19.3, 'bessellb', 'ab')
    else:
        model.set(amplitude=1.0)
        model.set_source_peakabsmag(-17.5, 'bessellb', 'ab')
    rows = []
    for band, sb in (('g', 'ztfg'), ('r', 'ztfr')):
        times = np.arange(59990.0, 60070.0, 4.0)
        flux = model.bandflux(sb, times, zp=31.4, zpsys='ab')
        keep = flux > 0
        times, flux = times[keep], flux[keep]
        err = np.maximum(0.02 * flux, 1.0)
        rows.append(pd.DataFrame({
            'mjd': times,
            'flux': flux + rng.normal(0, err),
            'flux_err': err,
            'band': band,
            'survey': 'ZTF',
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# No-sncosmo paths (always run)
# ---------------------------------------------------------------------------

def test_fit_template_without_sncosmo(monkeypatch):
    monkeypatch.setattr(pf, 'HAS_SNCOSMO', False)
    res = fit_template(pd.DataFrame(), 'nugent-sn2p')
    assert res['status'] == 'sncosmo_not_installed'


def test_tournament_without_sncosmo(monkeypatch):
    monkeypatch.setattr(pf, 'HAS_SNCOSMO', False)
    res = run_template_tournament(pd.DataFrame())
    assert res['status'] == 'sncosmo_not_installed'


def test_fit_template_insufficient_data():
    if not pf.HAS_SNCOSMO:
        pytest.skip('sncosmo not installed')
    lc = pd.DataFrame({'mjd': [60000.0, 60001.0], 'flux': [10.0, 11.0],
                       'flux_err': [1.0, 1.0], 'band': ['g', 'g'],
                       'survey': ['ZTF', 'ZTF']})
    res = fit_template(lc, 'nugent-sn2p')
    assert res['status'] == 'insufficient_data'


def test_cc_templates_registry():
    assert set(CC_TEMPLATES) == {'Ibc', 'IIP', 'IIn'}
    assert all(v.startswith('nugent-') for v in CC_TEMPLATES.values())


# ---------------------------------------------------------------------------
# Fitting paths (skip when sources unavailable / offline)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _sources_available('nugent-sn2p'),
                    reason='nugent-sn2p not available (offline?)')
def test_fit_template_recovers_iip():
    lc = _synthesize_lc('nugent-sn2p', z=0.05)
    res = fit_template(lc, 'nugent-sn2p', z=0.05)
    assert res['status'] == 'ok'
    assert res['chi2_dof'] < 3.0
    assert np.isfinite(res['peak_mjd'])
    assert np.isfinite(res['peak_mag_obs'])


@pytest.mark.skipif(
    not _sources_available('salt2-extended', *CC_TEMPLATES.values()),
    reason='tournament sources not available (offline?)')
def test_tournament_types_a_iip_as_non_ia():
    lc = _synthesize_lc('nugent-sn2p', z=0.05)
    res = run_template_tournament(lc, z=0.05)
    assert res['status'] == 'ok'
    assert res['template_best'] in ('IIP', 'IIn')   # II flavors are close kin
    assert res['template_chi2s']['IIP'] < res['template_chi2s'].get(
        'Ia', np.inf)
    assert np.isfinite(res['template_peak_mjd'])


@pytest.mark.skipif(
    not _sources_available('salt2-extended', *CC_TEMPLATES.values()),
    reason='tournament sources not available (offline?)')
def test_tournament_types_an_ia_as_ia():
    lc = _synthesize_lc('salt2-extended', z=0.05)
    res = run_template_tournament(lc, z=0.05)
    assert res['status'] == 'ok'
    assert res['template_best'] == 'Ia'
    assert np.isfinite(res['template_margin'])


@pytest.mark.skipif(
    not _sources_available('salt2-extended', *CC_TEMPLATES.values()),
    reason='tournament sources not available (offline?)')
def test_tournament_reuses_existing_salt_result():
    lc = _synthesize_lc('salt2-extended', z=0.05)
    salt = pf.fit_salt(lc, z=0.05)
    assert salt['status'] == 'ok'
    res = run_template_tournament(lc, z=0.05, salt=salt)
    assert res['status'] == 'ok'
    assert res['template_chi2s']['Ia'] == pytest.approx(salt['chi2_dof'])


# ---------------------------------------------------------------------------
# Pipeline enrichment wrapper
# ---------------------------------------------------------------------------

def test_enrich_finalist_typing_handles_missing_lcs():
    import run_tonight as rt
    summary = pd.DataFrame({
        'diaObjectId': ['a', 'b'],
        'redshift': [np.nan, 0.05],
        'z_source': ['none', 'tns_specz'],
        'ned_name': [np.nan, np.nan],
    })
    out = rt.enrich_finalist_typing(summary.copy(), fit_results={})
    if not pf.HAS_SNCOSMO:
        assert 'template_best' not in out.columns
        return
    # Columns exist, nothing fit (no light curves) — all null, no raise
    assert 'template_best' in out.columns
    assert out['template_best'].isna().all()
    assert out['template_best_chi2'].isna().all()
