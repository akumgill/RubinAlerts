"""Light curve peak fitting for SN Ia candidates.

Provides two fitting methods:
1. Inverted parabola (per-band) — always available, uses scipy.optimize.curve_fit
2. SALT2/SALT3 template (multi-band) — optional, requires sncosmo

All fitting is done in flux space (nanoJanskys) to handle negative
difference-imaging values naturally.
"""

import logging
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# AB magnitude zeropoint for nanoJansky flux: mag = -2.5 * log10(flux_nJy) + 31.4
AB_ZP_NJY = 31.4

# Band priority for selecting the "best" headline estimate.
# r-band is closest to rest-frame B for typical DDF SN Ia redshifts.
BAND_PRIORITY = ['r', 'g', 'i', 'z', 'y', 'u']

# Check for optional sncosmo
try:
    import sncosmo
    HAS_SNCOSMO = True
except ImportError:
    HAS_SNCOSMO = False

# LSST band name mapping for sncosmo
LSST_BAND_MAP = {
    'u': 'lsstu', 'g': 'lsstg', 'r': 'lsstr',
    'i': 'lssti', 'z': 'lsstz', 'y': 'lssty',
}


# ---------------------------------------------------------------------------
# Parabola model
# ---------------------------------------------------------------------------

def _inverted_parabola(t, peak_flux, t0, a):
    """Inverted parabola in flux space: flux(t) = peak_flux - a * (t - t0)^2.

    Parameters
    ----------
    t : array-like
        Time (MJD).
    peak_flux : float
        Flux at peak (nJy).
    t0 : float
        Time of peak (MJD).
    a : float
        Curvature (must be > 0 for a peak).
    """
    return peak_flux - a * (t - t0) ** 2


def fit_parabola_single_band(mjd, flux, flux_err, band_name='?'):
    """Fit an inverted parabola to a single band's flux light curve.

    Parameters
    ----------
    mjd : array-like
        Observation times (MJD).
    flux : array-like
        Flux values (nJy, may include negatives).
    flux_err : array-like
        Flux uncertainties (nJy).
    band_name : str
        Band label for logging.

    Returns
    -------
    dict with keys: peak_mjd, peak_flux, peak_mag, peak_mag_err,
                    curvature, chi2_dof, n_points, band, status
    """
    mjd = np.asarray(mjd, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    # Remove NaN / inf
    good = np.isfinite(mjd) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    mjd, flux, flux_err = mjd[good], flux[good], flux_err[good]

    n = len(mjd)
    result = {
        'band': band_name,
        'n_points': n,
        'peak_mjd': np.nan,
        'peak_flux': np.nan,
        'peak_mag': np.nan,
        'peak_mag_err': np.nan,
        'curvature': np.nan,
        'chi2_dof': np.nan,
        'status': 'insufficient_data',
    }

    if n < 3:
        return result

    # Initial guesses
    idx_max = np.argmax(flux)
    p0 = [flux[idx_max], mjd[idx_max], 1.0]

    try:
        popt, pcov = curve_fit(
            _inverted_parabola, mjd, flux,
            p0=p0, sigma=flux_err, absolute_sigma=True,
            maxfev=5000,
        )
    except (RuntimeError, ValueError) as e:
        logger.debug("Parabola fit failed for band %s: %s", band_name, e)
        result['status'] = 'fit_failed'
        return result

    peak_flux, t0, a = popt
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan] * 3

    # Chi-squared
    residuals = flux - _inverted_parabola(mjd, *popt)
    chi2 = np.sum((residuals / flux_err) ** 2)
    dof = max(n - 3, 1)
    chi2_dof = chi2 / dof

    # Magnitude conversion (only if peak flux is positive)
    if peak_flux > 0:
        peak_mag = -2.5 * np.log10(peak_flux) + AB_ZP_NJY
        # Error propagation: d(mag)/d(flux) = -2.5 / (ln10 * flux)
        peak_flux_err = perr[0]
        peak_mag_err = (2.5 / np.log(10)) * (peak_flux_err / peak_flux) if peak_flux_err > 0 else np.nan
    else:
        peak_mag = np.nan
        peak_mag_err = np.nan

    # Status logic
    if n == 3:
        status = 'underdetermined'
    elif chi2_dof > 10:
        status = 'poor_fit'
    else:
        status = 'ok'

    result.update({
        'peak_mjd': t0,
        'peak_flux': peak_flux,
        'peak_mag': peak_mag,
        'peak_mag_err': peak_mag_err,
        'curvature': a,
        'chi2_dof': chi2_dof,
        'status': status,
    })
    return result


# ---------------------------------------------------------------------------
# Multi-band parabola wrapper
# ---------------------------------------------------------------------------

def fit_parabola(lc_df, bands=None):
    """Fit inverted parabola to each band independently.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve with columns: mjd, psfFlux (or flux), psfFluxErr (or flux_err),
        band_name (or band).
    bands : list of str, optional
        Bands to fit. If None, fits all available bands.

    Returns
    -------
    dict with keys: per_band, best, method
    """
    # Normalise column names
    df = lc_df.copy()
    if 'psfFlux' in df.columns and 'flux' not in df.columns:
        df['flux'] = df['psfFlux']
    if 'psfFluxErr' in df.columns and 'flux_err' not in df.columns:
        df['flux_err'] = df['psfFluxErr']
    if 'band_name' in df.columns and 'band' not in df.columns:
        df['band'] = df['band_name']

    if 'flux' not in df.columns or 'flux_err' not in df.columns:
        logger.warning("Light curve missing flux/flux_err columns")
        return {'per_band': {}, 'best': None, 'method': 'parabola'}

    available_bands = df['band'].dropna().unique().tolist() if 'band' in df.columns else []
    if bands is None:
        bands = available_bands

    per_band = {}
    for b in bands:
        mask = df['band'] == b
        if mask.sum() == 0:
            continue
        sub = df[mask]
        result = fit_parabola_single_band(
            sub['mjd'].values, sub['flux'].values, sub['flux_err'].values,
            band_name=b,
        )
        per_band[b] = result

    # Pick best band by priority
    best = None
    for b in BAND_PRIORITY:
        if b in per_band and per_band[b]['status'] in ('ok', 'underdetermined'):
            best = per_band[b]
            break

    # Fallback: any band with a successful fit
    if best is None:
        for b in BAND_PRIORITY:
            if b in per_band and per_band[b]['status'] != 'insufficient_data':
                best = per_band[b]
                break

    return {'per_band': per_band, 'best': best, 'method': 'parabola'}


# ---------------------------------------------------------------------------
# Optional SALT2 / SALT3 fit
# ---------------------------------------------------------------------------

def fit_salt(lc_df, model_name='salt2', z=None, z_range=(0.01, 1.2)):
    """Multi-band SN Ia template fit using sncosmo.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve with mjd, psfFlux/flux, psfFluxErr/flux_err, band_name/band.
    model_name : str
        'salt2' or 'salt3'.
    z : float, optional
        Fixed redshift. If None, z floats within z_range.
    z_range : tuple
        (z_min, z_max) bounds when z is free.

    Returns
    -------
    dict with fit parameters or status='unavailable'/'failed'.
    """
    if not HAS_SNCOSMO:
        return {'status': 'sncosmo_not_installed', 'method': 'salt'}

    from astropy.table import Table

    df = lc_df.copy()
    if 'psfFlux' in df.columns and 'flux' not in df.columns:
        df['flux'] = df['psfFlux']
    if 'psfFluxErr' in df.columns and 'flux_err' not in df.columns:
        df['flux_err'] = df['psfFluxErr']
    if 'band_name' in df.columns and 'band' not in df.columns:
        df['band'] = df['band_name']

    # Filter to LSST bands that sncosmo knows about
    df = df[df['band'].isin(LSST_BAND_MAP)].copy()
    df['sncosmo_band'] = df['band'].map(LSST_BAND_MAP)

    good = df['flux'].notna() & df['flux_err'].notna() & (df['flux_err'] > 0) & df['mjd'].notna()
    df = df[good]

    n_points = len(df)
    n_bands = df['band'].nunique()

    if n_points < 5 or n_bands < 2:
        return {
            'status': 'insufficient_data',
            'method': 'salt',
            'n_points': n_points,
            'n_bands': n_bands,
        }

    # Build astropy Table for sncosmo
    data = Table({
        'time': df['mjd'].values,
        'band': df['sncosmo_band'].values,
        'flux': df['flux'].values,
        'fluxerr': df['flux_err'].values,
        'zp': np.full(n_points, AB_ZP_NJY),
        'zpsys': np.full(n_points, 'ab', dtype='U2'),
    })

    try:
        model = sncosmo.Model(source=model_name)

        params = ['t0', 'x0', 'x1', 'c']
        bounds = {
            't0': (df['mjd'].min() - 20, df['mjd'].max() + 20),
            'x1': (-5.0, 5.0),
            'c': (-0.5, 0.5),
        }

        if z is not None:
            model.set(z=z)
        else:
            params.insert(0, 'z')
            bounds['z'] = z_range

        result, fitted_model = sncosmo.fit_lc(
            data, model, params, bounds=bounds,
        )

        # Peak magnitude in rest-frame B
        try:
            peak_mag_B = fitted_model.source_peakmag('bessellb', 'ab')
        except Exception:
            peak_mag_B = np.nan

        return {
            'status': 'ok',
            'method': 'salt',
            't0': result.parameters[result.param_names.index('t0')],
            'x0': result.parameters[result.param_names.index('x0')],
            'x1': result.parameters[result.param_names.index('x1')],
            'c': result.parameters[result.param_names.index('c')],
            'z': fitted_model.get('z'),
            'peak_mag_B': peak_mag_B,
            'chi2': result.chisq,
            'ndof': result.ndof,
            'chi2_dof': result.chisq / max(result.ndof, 1),
            'n_points': n_points,
            'n_bands': n_bands,
        }

    except Exception as e:
        logger.debug("SALT fit failed: %s", e)
        return {
            'status': 'fit_failed',
            'method': 'salt',
            'error': str(e),
            'n_points': n_points,
            'n_bands': n_bands,
        }


# ---------------------------------------------------------------------------
# PeakFitter: batch orchestrator
# ---------------------------------------------------------------------------

class PeakFitter:
    """Batch peak-fitting orchestrator.

    Parameters
    ----------
    monitor : SupernovaMonitor
        Instance used to retrieve light curves.
    """

    def __init__(self, monitor):
        self.monitor = monitor

    def fit_candidate(self, object_id, broker='ALeRCE-LSST',
                      use_salt=False, z=None):
        """Fit peak for one candidate.

        Returns
        -------
        dict with keys: object_id, parabola, salt (if requested).
        """
        lc = self.monitor.get_light_curve(object_id, broker=broker)

        result = {'object_id': object_id, 'broker': broker}

        if lc is None or len(lc) == 0:
            result['parabola'] = {'per_band': {}, 'best': None, 'method': 'parabola'}
            if use_salt:
                result['salt'] = {'status': 'no_data', 'method': 'salt'}
            return result

        result['parabola'] = fit_parabola(lc)

        if use_salt:
            result['salt'] = fit_salt(lc, z=z)

        return result

    def fit_all_candidates(self, candidates_df, broker='ALeRCE-LSST',
                           id_column='object_id_ALeRCE-LSST',
                           use_salt=False, max_candidates=None):
        """Fit peak for all candidates with valid IDs.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            Candidates table from the pipeline.
        broker : str
            Broker to use for light curve retrieval.
        id_column : str
            Column name containing the object IDs for this broker.
        use_salt : bool
            Whether to attempt SALT2 fit.
        max_candidates : int, optional
            Limit number of candidates to fit (for testing).

        Returns
        -------
        dict of {object_id: fit_result}
        """
        if id_column not in candidates_df.columns:
            logger.warning("Column '%s' not found in candidates", id_column)
            return {}

        ids = candidates_df[id_column].dropna()
        ids = ids[ids.astype(str) != 'nan']

        if max_candidates is not None:
            ids = ids.head(max_candidates)

        logger.info("Fitting %d candidates from %s...", len(ids), broker)

        fit_results = {}
        for i, oid in enumerate(ids):
            try:
                fit_results[oid] = self.fit_candidate(
                    oid, broker=broker, use_salt=use_salt,
                )
                status = 'no data'
                best = fit_results[oid]['parabola'].get('best')
                if best is not None:
                    status = f"{best['band']}: {best['status']}"
                logger.debug("[%d/%d] %s — %s", i + 1, len(ids), oid, status)
            except Exception as e:
                logger.warning("Error fitting %s: %s", oid, e)
                fit_results[oid] = {
                    'object_id': oid,
                    'broker': broker,
                    'parabola': {'per_band': {}, 'best': None, 'method': 'parabola'},
                }

        n_ok = sum(
            1 for r in fit_results.values()
            if r['parabola'].get('best') and r['parabola']['best']['status'] == 'ok'
        )
        logger.info("Fitting complete: %d/%d with status 'ok'", n_ok, len(fit_results))
        return fit_results

    @staticmethod
    def get_summary_table(fit_results):
        """Convert fit results dict to a summary DataFrame.

        Returns
        -------
        pd.DataFrame with columns: object_id, n_points, best_band, peak_mjd,
            peak_flux, peak_mag, peak_mag_err, chi2_dof, fit_status, bands_ok,
            (+ salt_t0, salt_x1, salt_c, salt_peak_mag_B, salt_chi2_dof,
              salt_status if SALT results present).
        """
        rows = []
        for oid, res in fit_results.items():
            best = res['parabola'].get('best')
            per_band = res['parabola'].get('per_band', {})

            bands_ok = [b for b, r in per_band.items()
                        if r['status'] in ('ok', 'underdetermined')]

            row = {
                'object_id': oid,
                'n_points': best['n_points'] if best else 0,
                'best_band': best['band'] if best else '',
                'peak_mjd': best['peak_mjd'] if best else np.nan,
                'peak_flux': best['peak_flux'] if best else np.nan,
                'peak_mag': best['peak_mag'] if best else np.nan,
                'peak_mag_err': best['peak_mag_err'] if best else np.nan,
                'chi2_dof': best['chi2_dof'] if best else np.nan,
                'fit_status': best['status'] if best else 'no_data',
                'bands_ok': ', '.join(bands_ok),
            }

            # SALT columns
            salt = res.get('salt')
            if salt is not None:
                row['salt_status'] = salt.get('status', '')
                if salt.get('status') == 'ok':
                    row['salt_t0'] = salt.get('t0', np.nan)
                    row['salt_x1'] = salt.get('x1', np.nan)
                    row['salt_c'] = salt.get('c', np.nan)
                    row['salt_peak_mag_B'] = salt.get('peak_mag_B', np.nan)
                    row['salt_chi2_dof'] = salt.get('chi2_dof', np.nan)
                    row['salt_z'] = salt.get('z', np.nan)

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def merge_peak_fits(candidates_df, fit_results, id_column='object_id_ALeRCE-LSST'):
        """Merge headline peak estimates into the candidates DataFrame.

        Adds columns: peak_mjd, peak_mag, peak_band, peak_fit_status.
        """
        peak_data = {}
        for oid, res in fit_results.items():
            best = res['parabola'].get('best')
            if best is not None:
                peak_data[oid] = {
                    'peak_mjd': best['peak_mjd'],
                    'peak_mag': best['peak_mag'],
                    'peak_band': best['band'],
                    'peak_fit_status': best['status'],
                }
            else:
                peak_data[oid] = {
                    'peak_mjd': np.nan,
                    'peak_mag': np.nan,
                    'peak_band': '',
                    'peak_fit_status': 'no_data',
                }

        peak_df = pd.DataFrame.from_dict(peak_data, orient='index')
        peak_df.index.name = id_column

        out = candidates_df.copy()
        # Drop existing peak columns to avoid conflicts
        for col in ['peak_mjd', 'peak_mag', 'peak_band', 'peak_fit_status']:
            if col in out.columns:
                out = out.drop(columns=[col])

        out = out.merge(peak_df, left_on=id_column, right_index=True, how='left')
        return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_parabola_fit(lc_df, fit_result, object_id='', figsize=(12, 6)):
    """Plot flux light curve with parabola overlay per band.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve data.
    fit_result : dict
        Output from fit_parabola().
    object_id : str
        Object label for plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    BAND_COLORS = {
        'u': 'violet', 'g': 'green', 'r': 'red',
        'i': 'goldenrod', 'z': 'purple', 'y': 'brown',
    }

    df = lc_df.copy()
    if 'psfFlux' in df.columns and 'flux' not in df.columns:
        df['flux'] = df['psfFlux']
    if 'psfFluxErr' in df.columns and 'flux_err' not in df.columns:
        df['flux_err'] = df['psfFluxErr']
    if 'band_name' in df.columns and 'band' not in df.columns:
        df['band'] = df['band_name']

    per_band = fit_result.get('per_band', {})
    bands = sorted(df['band'].dropna().unique())

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for b in bands:
        color = BAND_COLORS.get(b, 'gray')
        mask = df['band'] == b
        sub = df[mask]

        # Data points
        ax.errorbar(
            sub['mjd'], sub['flux'],
            yerr=sub.get('flux_err'),
            fmt='o', color=color, markersize=4, alpha=0.6,
            label=f'{b} data',
        )

        # Parabola overlay
        if b in per_band and per_band[b]['status'] not in ('insufficient_data', 'fit_failed'):
            r = per_band[b]
            t_grid = np.linspace(sub['mjd'].min(), sub['mjd'].max(), 200)
            flux_model = _inverted_parabola(t_grid, r['peak_flux'], r['peak_mjd'], r['curvature'])
            ax.plot(t_grid, flux_model, '-', color=color, alpha=0.8,
                    label=f'{b} fit ({r["status"]})')

            # Peak marker
            ax.plot(r['peak_mjd'], r['peak_flux'], '*', color=color,
                    markersize=14, markeredgecolor='black', markeredgewidth=0.5,
                    zorder=5)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('MJD')
    ax.set_ylabel('Flux (nJy)')
    ax.set_title(f'{object_id} — Parabola Peak Fit')
    ax.legend(fontsize=8, ncol=min(len(bands), 4), loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
