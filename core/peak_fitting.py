"""Light curve peak fitting for SN Ia candidates.

Provides three fitting methods:
1. Inverted parabola (per-band) — always available, uses scipy.optimize.curve_fit
2. SALT2/SALT3 template (multi-band) — optional, requires sncosmo
3. Villar SPM (per-band) — uses precomputed ALeRCE features, no fitting needed

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


def clean_light_curve(lc_df, sigma_clip=4.0, prefer_detection=True):
    """Clean a light curve for fitting.

    1. Deduplicate: when a DiaSource detection and forced photometry
       measurement share the same (MJD, band), keep the detection
       (it has better-characterized errors from the alert pipeline).
    2. Per-night sigma clipping: for each (night, band), reject points
       whose flux is >sigma_clip standard deviations from the weighted
       mean.  This removes catastrophic forced-phot outliers.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve with mjd, flux, flux_err, band, and optionally
        'source' columns.
    sigma_clip : float
        Rejection threshold in units of the point's own error bar.
    prefer_detection : bool
        If True, at duplicate MJDs keep the detection row.

    Returns
    -------
    Cleaned DataFrame (copy).
    """
    df = lc_df.copy()

    # Normalize column names
    if 'psfFlux' in df.columns and 'flux' not in df.columns:
        df['flux'] = df['psfFlux']
    if 'psfFluxErr' in df.columns and 'flux_err' not in df.columns:
        df['flux_err'] = df['psfFluxErr']
    if 'band_name' in df.columns and 'band' not in df.columns:
        df['band'] = df['band_name']

    n_before = len(df)

    # --- Step 1: Deduplicate detection + forced_phot at same MJD ---
    if prefer_detection and 'source' in df.columns:
        # Round MJD to 1e-3 days (~86 seconds) for matching
        df['_mjd_key'] = np.round(df['mjd'], 3)
        # Sort so detections come first (alphabetically 'detection' < 'forced_phot')
        df = df.sort_values(['_mjd_key', 'band', 'source'])
        # Keep first occurrence per (mjd_key, band) — this is the detection
        df = df.drop_duplicates(subset=['_mjd_key', 'band'], keep='first')
        df = df.drop(columns=['_mjd_key'])
        n_dedup = n_before - len(df)
        if n_dedup > 0:
            logger.debug("Deduplicated %d rows at shared MJDs", n_dedup)

    # --- Step 2: Per-night sigma clipping ---
    if sigma_clip > 0 and 'flux' in df.columns and 'flux_err' in df.columns:
        df = df.reset_index(drop=True)
        df['_night'] = np.floor(df['mjd']).astype(int)
        drop_idx = []

        for (night, band), grp in df.groupby(['_night', 'band']):
            if len(grp) < 3:
                continue  # too few to clip

            flux = grp['flux'].values.astype(float)
            err = grp['flux_err'].values.astype(float)
            good_err = (err > 0) & np.isfinite(err)

            if good_err.sum() < 3:
                continue

            # Weighted mean
            w = np.where(good_err, 1.0 / err ** 2, 0)
            wmean = np.sum(w * flux) / np.sum(w) if np.sum(w) > 0 else np.mean(flux)

            # Reject points far from weighted mean
            resid = np.abs(flux - wmean)
            threshold = sigma_clip * err
            bad = good_err & (resid > threshold)

            if bad.any():
                drop_idx.extend(grp.index[bad].tolist())

        if drop_idx:
            df = df.drop(index=drop_idx)
            logger.debug("Sigma-clipped %d outlier points (%.1f sigma)",
                        len(drop_idx), sigma_clip)
        df = df.drop(columns=['_night'])

    logger.debug("Light curve cleaned: %d → %d points", n_before, len(df))
    return df.reset_index(drop=True)

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


def fit_parabola_single_band(mjd, flux, flux_err, band_name='?', A_band=None):
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
    A_band : float, optional
        Galactic extinction A_SFD for this band (magnitudes).
        If provided, peak_mag_corrected = peak_mag - A_band is added.

    Returns
    -------
    dict with keys: peak_mjd, peak_flux, peak_mag, peak_mag_err,
                    peak_mag_corrected, A_band, curvature, chi2_dof,
                    n_points, band, status
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
        'peak_mag_corrected': np.nan,
        'A_band': A_band if A_band is not None else np.nan,
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

    # Extinction-corrected magnitude
    peak_mag_corrected = np.nan
    if not np.isnan(peak_mag) and A_band is not None and not np.isnan(A_band):
        peak_mag_corrected = peak_mag - A_band

    result.update({
        'peak_mjd': t0,
        'peak_flux': peak_flux,
        'peak_mag': peak_mag,
        'peak_mag_err': peak_mag_err,
        'peak_mag_corrected': peak_mag_corrected,
        'A_band': A_band if A_band is not None else np.nan,
        'curvature': a,
        'chi2_dof': chi2_dof,
        'status': status,
    })
    return result


# ---------------------------------------------------------------------------
# Multi-band parabola wrapper
# ---------------------------------------------------------------------------

def fit_parabola(lc_df, bands=None, extinction=None):
    """Fit inverted parabola to each band independently.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve with columns: mjd, psfFlux (or flux), psfFluxErr (or flux_err),
        band_name (or band).
    bands : list of str, optional
        Bands to fit. If None, fits all available bands.
    extinction : dict, optional
        Mapping of band letter to A_SFD value (e.g. {'g': 0.08, 'r': 0.06}).

    Returns
    -------
    dict with keys: per_band, best, method
    """
    # Clean light curve: deduplicate and sigma-clip
    df = clean_light_curve(lc_df)
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

    extinction = extinction or {}

    per_band = {}
    for b in bands:
        mask = df['band'] == b
        if mask.sum() == 0:
            continue
        sub = df[mask]
        a_band = extinction.get(b)
        result = fit_parabola_single_band(
            sub['mjd'].values, sub['flux'].values, sub['flux_err'].values,
            band_name=b, A_band=a_band,
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
# Villar SPM model (Villar et al. 2019)
# ---------------------------------------------------------------------------

# ALeRCE stores SPM_A normalized by 1e26. To convert to AB magnitudes:
# mag = -2.5 * log10(flux) + 2.5*26 - 48.6
VILLAR_FLUX_ZP = 2.5 * 26 - 48.6  # = 16.4

# ZTF filter ID → band letter
VILLAR_FID_MAP = {1: 'g', 2: 'r'}


def villar_flux(mjd, firstmjd, A, t0, beta, gamma, tau_rise, tau_fall):
    """Evaluate the Villar SPM (Supernova Parametric Model) in flux space.

    Parameters
    ----------
    mjd : array-like
        Observation times (MJD).
    firstmjd : float
        First detection MJD (t0 and gamma are relative to this).
    A : float
        Amplitude (in 1e26 flux units as stored by ALeRCE).
    t0 : float
        Explosion time offset from firstmjd (days).
    beta : float
        Linear decline fraction (0–1, dimensionless).
    gamma : float
        Time from explosion to peak (days).
    tau_rise : float
        Rise timescale (days).
    tau_fall : float
        Decline timescale (days).

    Returns
    -------
    array of flux values (in ALeRCE 1e26 units).
    """
    mjd = np.asarray(mjd, dtype=float)
    mjd0 = t0 + firstmjd       # explosion epoch
    mjd1 = mjd0 + gamma        # peak epoch

    F = np.zeros_like(mjd)
    rising = mjd < mjd1

    if rising.any():
        dt = mjd[rising] - mjd0
        den = 1.0 + np.exp(-dt / tau_rise)
        F[rising] = A * (1.0 - beta * dt / gamma) / den

    falling = ~rising
    if falling.any():
        dt_rise = mjd[falling] - mjd0
        dt_fall = mjd[falling] - mjd1
        den = 1.0 + np.exp(-dt_rise / tau_rise)
        F[falling] = A * (1.0 - beta) * np.exp(-dt_fall / tau_fall) / den

    return F


def _villar_nJy(mjd, A_nJy, t0_mjd, beta, gamma, tau_rise, tau_fall):
    """Villar model in nanoJansky flux space (for curve_fit).

    Unlike villar_flux(), here A_nJy is in nJy (not 1e26 units) and
    t0_mjd is the absolute explosion MJD (not offset from firstmjd).
    """
    mjd = np.asarray(mjd, dtype=float)
    mjd1 = t0_mjd + gamma  # peak epoch

    F = np.zeros_like(mjd)
    rising = mjd < mjd1

    if rising.any():
        dt = mjd[rising] - t0_mjd
        den = 1.0 + np.exp(-dt / tau_rise)
        F[rising] = A_nJy * (1.0 - beta * dt / gamma) / den

    falling = ~rising
    if falling.any():
        dt_rise = mjd[falling] - t0_mjd
        dt_fall = mjd[falling] - mjd1
        den = 1.0 + np.exp(-dt_rise / tau_rise)
        F[falling] = A_nJy * (1.0 - beta) * np.exp(-dt_fall / tau_fall) / den

    return F


def fit_villar_single_band(mjd, flux, flux_err, band_name='?', A_band=None):
    """Fit the Villar SPM model to a single band's flux light curve.

    Parameters
    ----------
    mjd, flux, flux_err : array-like
        Observations in nJy.
    band_name : str
        Band label.
    A_band : float, optional
        Galactic extinction for this band.

    Returns
    -------
    dict with: peak_mjd, peak_flux, peak_mag, peak_mag_err,
               peak_mag_corrected, A_band, params, chi2_dof,
               n_points, band, status
    """
    mjd = np.asarray(mjd, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    good = np.isfinite(mjd) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    mjd, flux, flux_err = mjd[good], flux[good], flux_err[good]

    n = len(mjd)
    result = {
        'band': band_name, 'n_points': n, 'method': 'villar',
        'peak_mjd': np.nan, 'peak_flux': np.nan,
        'peak_mag': np.nan, 'peak_mag_err': np.nan,
        'peak_mag_corrected': np.nan, 'A_band': A_band,
        'chi2_dof': np.nan, 'params': None, 'status': 'insufficient_data',
    }

    if n < 6:
        return result

    # Initial guesses from the data
    idx_max = np.argmax(flux)
    A0 = max(flux[idx_max], 100.0)
    t_peak0 = mjd[idx_max]
    t_first = mjd.min()
    gamma0 = max(t_peak0 - t_first, 3.0)
    t0_0 = t_peak0 - gamma0  # explosion time

    p0 = [A0, t0_0, 0.1, gamma0, 5.0, 20.0]
    # Bounds: A_nJy, t0_mjd, beta, gamma, tau_rise, tau_fall
    bounds_lo = [0.0, t_first - 60, 0.0, 1.0, 0.5, 1.0]
    bounds_hi = [A0 * 10, t_peak0 + 10, 0.99, 120.0, 60.0, 200.0]

    try:
        popt, pcov = curve_fit(
            _villar_nJy, mjd, flux, p0=p0, sigma=flux_err,
            absolute_sigma=True, maxfev=10000,
            bounds=(bounds_lo, bounds_hi),
        )
        A_fit, t0_fit, beta_fit, gamma_fit, tau_rise_fit, tau_fall_fit = popt
        result['params'] = {
            'A_nJy': A_fit, 't0_mjd': t0_fit, 'beta': beta_fit,
            'gamma': gamma_fit, 'tau_rise': tau_rise_fit,
            'tau_fall': tau_fall_fit,
        }

        # Compute chi2
        model_flux = _villar_nJy(mjd, *popt)
        residuals = (flux - model_flux) / flux_err
        dof = max(n - 6, 1)
        chi2_dof = float(np.sum(residuals ** 2) / dof)
        result['chi2_dof'] = chi2_dof

        # Find peak numerically
        search_lo = t0_fit
        search_hi = t0_fit + gamma_fit + 3 * tau_fall_fit
        mjd_grid = np.linspace(search_lo, search_hi, 2000)
        flux_grid = _villar_nJy(mjd_grid, *popt)
        idx_peak = np.argmax(flux_grid)
        peak_flux = flux_grid[idx_peak]
        peak_mjd = mjd_grid[idx_peak]

        result['peak_mjd'] = peak_mjd
        result['peak_flux'] = peak_flux

        if peak_flux > 0:
            peak_mag = -2.5 * np.log10(peak_flux) + AB_ZP_NJY
            result['peak_mag'] = peak_mag
            if A_band is not None and np.isfinite(A_band):
                result['peak_mag_corrected'] = peak_mag - A_band

        # Status
        if chi2_dof <= 10:
            result['status'] = 'ok'
        else:
            result['status'] = 'poor_fit'

    except (RuntimeError, ValueError) as e:
        logger.debug("Villar fit failed for %s: %s", band_name, e)
        result['status'] = 'fit_failed'

    return result


def fit_villar(lc_df, bands=None, extinction=None):
    """Fit Villar SPM model to each band independently.

    Same interface as fit_parabola(), but fits the 6-parameter Villar
    model instead of an inverted parabola.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve with columns: mjd, flux, flux_err, band.
    bands : list of str, optional
        Bands to fit.
    extinction : dict, optional
        Band → A_SFD mapping.

    Returns
    -------
    dict with keys: per_band, best, method
    """
    df = clean_light_curve(lc_df)
    if 'psfFlux' in df.columns and 'flux' not in df.columns:
        df['flux'] = df['psfFlux']
    if 'psfFluxErr' in df.columns and 'flux_err' not in df.columns:
        df['flux_err'] = df['psfFluxErr']
    if 'band_name' in df.columns and 'band' not in df.columns:
        df['band'] = df['band_name']

    if 'flux' not in df.columns or 'flux_err' not in df.columns:
        logger.warning("Light curve missing flux/flux_err columns")
        return {'per_band': {}, 'best': None, 'method': 'villar'}

    available_bands = df['band'].dropna().unique().tolist() if 'band' in df.columns else []
    if bands is None:
        bands = available_bands

    extinction = extinction or {}

    per_band = {}
    for b in bands:
        mask = df['band'] == b
        if mask.sum() < 6:
            continue
        sub = df[mask]
        a_band = extinction.get(b)
        result = fit_villar_single_band(
            sub['mjd'].values, sub['flux'].values, sub['flux_err'].values,
            band_name=b, A_band=a_band,
        )
        per_band[b] = result

    # Pick best band by priority
    best = None
    for b in BAND_PRIORITY:
        if b in per_band and per_band[b]['status'] in ('ok',):
            best = per_band[b]
            break
    # Fall back to any ok fit
    if best is None:
        for b in per_band:
            if per_band[b]['status'] == 'ok':
                best = per_band[b]
                break

    return {'per_band': per_band, 'best': best, 'method': 'villar'}


def fit_villar_multiband(lc_df, bands=None, extinction=None, min_points_per_band=5):
    """Multi-band Villar SPM fit with shared explosion epoch.

    Fits all bands simultaneously with a single shared t0 (explosion
    time) while each band gets its own A, beta, gamma, tau_rise,
    tau_fall.  This produces much more robust results than independent
    per-band fits because the explosion epoch is constrained by data
    from every band.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve with columns: mjd, flux, flux_err, band.
    bands : list of str, optional
        Bands to include.  Default: all bands with enough data.
    extinction : dict, optional
        Band → A_SFD mapping.
    min_points_per_band : int
        Minimum observations required to include a band.

    Returns
    -------
    dict with keys: per_band, best, method, shared_t0, n_bands_fit
    """
    from scipy.optimize import least_squares

    # Clean light curve: deduplicate and sigma-clip
    df = clean_light_curve(lc_df)

    if 'flux' not in df.columns or 'flux_err' not in df.columns:
        logger.warning("Light curve missing flux/flux_err columns")
        return {'per_band': {}, 'best': None, 'method': 'villar_multiband'}

    # Remove NaN / non-positive errors
    good = (df['flux'].notna() & df['flux_err'].notna() &
            np.isfinite(df['flux']) & np.isfinite(df['flux_err']) &
            (df['flux_err'] > 0) & df['mjd'].notna())
    df = df[good].copy()

    available = df['band'].value_counts()
    if bands is None:
        bands = [b for b in BAND_PRIORITY if b in available and available[b] >= min_points_per_band]
        # Also include any band not in priority list
        for b in available.index:
            if b not in bands and available[b] >= min_points_per_band:
                bands.append(b)

    if len(bands) == 0:
        return {'per_band': {}, 'best': None, 'method': 'villar_multiband',
                'shared_t0': np.nan, 'n_bands_fit': 0}

    df = df[df['band'].isin(bands)].copy().reset_index(drop=True)

    # Index mapping: which rows belong to which band (positional, 0-based)
    band_indices = {}
    for b in bands:
        band_indices[b] = df.index[df['band'] == b].tolist()

    mjd_all = df['mjd'].values.astype(float)
    flux_all = df['flux'].values.astype(float)
    err_all = df['flux_err'].values.astype(float)

    n_bands = len(bands)
    # Parameter layout: [t0, A_0, beta_0, gamma_0, tau_rise_0, tau_fall_0,
    #                         A_1, beta_1, gamma_1, tau_rise_1, tau_fall_1, ...]
    # Total: 1 + 5*n_bands parameters

    # Initial guesses from per-band data
    idx_max_global = np.argmax(flux_all)
    t_peak_global = mjd_all[idx_max_global]
    t_first = mjd_all.min()
    gamma_guess = max(t_peak_global - t_first, 3.0)
    t0_guess = t_peak_global - gamma_guess

    p0 = [t0_guess]
    lo = [t_first - 60]
    hi = [t_peak_global + 10]

    for b in bands:
        idx = band_indices[b]
        b_flux = flux_all[idx]
        b_mjd = mjd_all[idx]
        idx_max = np.argmax(b_flux)
        A0 = max(b_flux[idx_max], 100.0)
        t_peak_b = b_mjd[idx_max]
        gamma_b = max(t_peak_b - t0_guess, 3.0)

        p0.extend([A0, 0.1, gamma_b, 5.0, 20.0])
        lo.extend([0.0, 0.0, 1.0, 0.5, 1.0])
        hi.extend([A0 * 10, 0.99, 120.0, 60.0, 200.0])

    p0 = np.array(p0)
    lo = np.array(lo)
    hi = np.array(hi)

    # Clip p0 to bounds
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)

    def residual_func(params):
        t0 = params[0]
        resid = np.empty(len(mjd_all))
        for i, b in enumerate(bands):
            offset = 1 + 5 * i
            A_nJy = params[offset]
            beta = params[offset + 1]
            gamma = params[offset + 2]
            tau_rise = params[offset + 3]
            tau_fall = params[offset + 4]

            idx = band_indices[b]
            mjd_b = mjd_all[idx]
            err_b = err_all[idx]

            model = _villar_nJy(mjd_b, A_nJy, t0, beta, gamma, tau_rise, tau_fall)
            resid[idx] = (flux_all[idx] - model) / err_b

        return resid

    try:
        result = least_squares(
            residual_func, p0, bounds=(lo, hi),
            method='trf', max_nfev=20000,
            loss='soft_l1',  # robust to outliers
        )
        popt = result.x
        cost = result.cost
        n_data = len(mjd_all)
        n_params = len(popt)
        dof = max(n_data - n_params, 1)
        chi2_dof = 2.0 * cost / dof  # least_squares cost = 0.5 * sum(resid^2) for linear loss

        shared_t0 = popt[0]
    except Exception as e:
        logger.warning("Multi-band Villar fit failed: %s", e)
        return {'per_band': {}, 'best': None, 'method': 'villar_multiband',
                'shared_t0': np.nan, 'n_bands_fit': 0}

    extinction = extinction or {}
    per_band = {}
    for i, b in enumerate(bands):
        offset = 1 + 5 * i
        A_nJy = popt[offset]
        beta = popt[offset + 1]
        gamma = popt[offset + 2]
        tau_rise = popt[offset + 3]
        tau_fall = popt[offset + 4]

        params = {
            'A_nJy': A_nJy, 't0_mjd': shared_t0,
            'beta': beta, 'gamma': gamma,
            'tau_rise': tau_rise, 'tau_fall': tau_fall,
        }

        # Per-band chi2
        idx = band_indices[b]
        mjd_b = mjd_all[idx]
        model_b = _villar_nJy(mjd_b, A_nJy, shared_t0, beta, gamma, tau_rise, tau_fall)
        resid_b = (flux_all[idx] - model_b) / err_all[idx]
        dof_b = max(len(idx) - 5, 1)
        chi2_b = float(np.sum(resid_b ** 2) / dof_b)

        # Find peak numerically
        search_hi = shared_t0 + gamma + 3 * tau_fall
        mjd_grid = np.linspace(shared_t0, search_hi, 2000)
        flux_grid = _villar_nJy(mjd_grid, A_nJy, shared_t0, beta, gamma, tau_rise, tau_fall)
        idx_peak = np.argmax(flux_grid)
        peak_flux = flux_grid[idx_peak]
        peak_mjd = mjd_grid[idx_peak]

        a_band = extinction.get(b)
        peak_mag = np.nan
        peak_mag_corrected = np.nan
        if peak_flux > 0:
            peak_mag = -2.5 * np.log10(peak_flux) + AB_ZP_NJY
            if a_band is not None and np.isfinite(a_band):
                peak_mag_corrected = peak_mag - a_band

        status = 'ok' if chi2_b <= 10 else 'poor_fit'

        per_band[b] = {
            'band': b, 'method': 'villar_multiband',
            'n_points': len(idx),
            'peak_mjd': peak_mjd, 'peak_flux': peak_flux,
            'peak_mag': peak_mag, 'peak_mag_err': np.nan,
            'peak_mag_corrected': peak_mag_corrected,
            'A_band': a_band,
            'chi2_dof': chi2_b,
            'params': params,
            'status': status,
        }

    # Pick best band
    best = None
    for b in BAND_PRIORITY:
        if b in per_band and per_band[b]['status'] == 'ok':
            best = per_band[b]
            break
    if best is None:
        for b in per_band:
            if per_band[b]['status'] == 'ok':
                best = per_band[b]
                break

    logger.info("Multi-band Villar: %d bands, shared t0=%.1f, chi2/dof=%.2f",
                n_bands, shared_t0, chi2_dof)

    return {
        'per_band': per_band, 'best': best,
        'method': 'villar_multiband',
        'shared_t0': shared_t0,
        'n_bands_fit': n_bands,
        'chi2_dof_global': chi2_dof,
    }


def villar_peak_from_params(firstmjd, A, t0, gamma, beta, tau_rise, tau_fall=None, chi=None):
    """Extract peak time and magnitude from Villar SPM parameters.

    Finds the true flux maximum numerically by evaluating the model on a
    fine grid and locating where the flux peaks.  This is more accurate
    than the analytic formula ``peak_mjd = firstmjd + t0 + gamma`` because
    the sigmoid rise term shifts the actual maximum.

    Parameters
    ----------
    firstmjd : float
        First detection MJD.
    A, t0, gamma, beta, tau_rise : float
        SPM parameters from ALeRCE feature table.
    tau_fall : float, optional
        Decline timescale. If None, defaults to tau_rise.
    chi : float, optional
        SPM goodness of fit (chi-squared).

    Returns
    -------
    dict with peak_mjd, peak_mag, villar_chi, status.
    """
    result = {
        'peak_mjd': np.nan,
        'peak_mag': np.nan,
        'villar_chi': chi if chi is not None else np.nan,
        'status': 'no_params',
    }

    # Validate parameters
    if any(np.isnan(v) for v in [firstmjd, A, t0, gamma]):
        return result

    if A <= 0 or gamma <= 0:
        result['status'] = 'invalid_params'
        return result

    if tau_fall is None:
        tau_fall = tau_rise

    # Numerically find the flux maximum on a fine grid around the
    # analytic transition point (firstmjd + t0 + gamma).
    mjd_approx = firstmjd + t0 + gamma
    search_lo = firstmjd + t0 - 2 * tau_rise
    search_hi = mjd_approx + 3 * tau_fall
    mjd_grid = np.linspace(search_lo, search_hi, 2000)

    flux_grid = villar_flux(mjd_grid, firstmjd,
                            A, t0, beta, gamma, tau_rise, tau_fall)

    idx_max = np.argmax(flux_grid)
    peak_flux = flux_grid[idx_max]
    peak_mjd = mjd_grid[idx_max]

    result['peak_mjd'] = peak_mjd

    if peak_flux > 0:
        result['peak_mag'] = -2.5 * np.log10(peak_flux) + VILLAR_FLUX_ZP
        result['status'] = 'ok'
    else:
        result['status'] = 'negative_flux'

    return result


def extract_villar_peaks(features_df, firstmjd_lookup):
    """Extract peak estimates from ALeRCE SPM features for multiple objects.

    Parameters
    ----------
    features_df : pd.DataFrame
        Pivoted SPM features from AlerceDBClient.query_features().
        Columns like: oid, SPM_A_1, SPM_A_2, SPM_t0_1, SPM_gamma_1, etc.
    firstmjd_lookup : dict
        Mapping of oid → firstMJD.

    Returns
    -------
    dict of {oid: {band: villar_result_dict}}
    """
    results = {}

    for _, row in features_df.iterrows():
        oid = row.get('oid')
        if not oid:
            continue

        firstmjd = firstmjd_lookup.get(oid)
        if firstmjd is None or np.isnan(firstmjd):
            continue

        oid_results = {}
        for fid, band in VILLAR_FID_MAP.items():
            sfx = f'_{fid}'
            A = row.get(f'SPM_A{sfx}', np.nan)
            t0 = row.get(f'SPM_t0{sfx}', np.nan)
            beta = row.get(f'SPM_beta{sfx}', np.nan)
            gamma = row.get(f'SPM_gamma{sfx}', np.nan)
            tau_rise = row.get(f'SPM_tau_rise{sfx}', np.nan)
            tau_fall = row.get(f'SPM_tau_fall{sfx}', np.nan)
            chi = row.get(f'SPM_chi{sfx}', np.nan)

            if not np.isnan(A):
                peak = villar_peak_from_params(
                    firstmjd, A, t0, gamma, beta, tau_rise,
                    tau_fall=tau_fall, chi=chi,
                )
                peak['band'] = band
                oid_results[band] = peak

        if oid_results:
            results[oid] = oid_results

    return results


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
                      use_salt=False, z=None, extinction=None):
        """Fit peak for one candidate.

        Parameters
        ----------
        object_id : str
            Object identifier for light curve retrieval.
        broker : str
            Broker to query for light curve.
        use_salt : bool
            Whether to attempt SALT2 fit.
        z : float, optional
            Fixed redshift for SALT fit.
        extinction : dict, optional
            Mapping of band letter to A_SFD value.

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

        result['parabola'] = fit_parabola(lc, extinction=extinction)

        if use_salt:
            result['salt'] = fit_salt(lc, z=z)

        # Villar params are added externally via enrich_with_villar()
        return result

    def fit_candidate_fink(self, fink_dia_object_id, extinction=None,
                           use_salt=False, z=None):
        """Fit peak using Fink LSST prompt photometry.

        Runs both parabola and Villar SPM fits on the Fink light curve.

        Parameters
        ----------
        fink_dia_object_id : str
            Fink diaObjectId for Rubin light curve retrieval.
        extinction : dict, optional
            Band → A_SFD mapping for extinction correction.
        use_salt : bool
            Whether to attempt SALT2 fit.
        z : float, optional
            Fixed redshift for SALT fit.

        Returns
        -------
        dict with keys: object_id, broker, parabola, villar,
                        light_curve, salt (if requested).
        """
        if self.monitor.fink_client is None:
            logger.warning("Fink client not available")
            return {
                'object_id': fink_dia_object_id,
                'broker': 'Fink-LSST',
                'parabola': {'per_band': {}, 'best': None, 'method': 'parabola'},
                'villar': {'per_band': {}, 'best': None, 'method': 'villar'},
            }

        lc = self.monitor.fink_client.get_light_curve(
            str(fink_dia_object_id), include_forced=True
        )

        result = {'object_id': fink_dia_object_id, 'broker': 'Fink-LSST'}

        if lc is None or len(lc) == 0:
            result['parabola'] = {'per_band': {}, 'best': None, 'method': 'parabola'}
            result['villar'] = {'per_band': {}, 'best': None, 'method': 'villar'}
            if use_salt:
                result['salt'] = {'status': 'no_data', 'method': 'salt'}
            return result

        result['parabola'] = fit_parabola(lc, extinction=extinction)
        result['villar'] = fit_villar_multiband(lc, extinction=extinction)
        result['light_curve'] = lc  # cache for plotting

        if use_salt:
            result['salt'] = fit_salt(lc, z=z)

        return result

    def fit_all_candidates_fink(self, candidates_df,
                                max_candidates=None, use_salt=False):
        """Fit peaks for candidates using Fink LSST prompt photometry.

        Requires candidates_df to have 'fink_diaObjectId' column
        (from FinkLSSTClient.crossmatch_candidates()).

        Parameters
        ----------
        candidates_df : pd.DataFrame
            Must have 'fink_diaObjectId' and optionally A_g, A_r, etc.
        max_candidates : int, optional
            Limit for testing.
        use_salt : bool
            Whether to attempt SALT2 fit.

        Returns
        -------
        dict of {fink_diaObjectId: fit_result}
        """
        if 'fink_diaObjectId' not in candidates_df.columns:
            logger.warning("No 'fink_diaObjectId' column — run Fink crossmatch first")
            return {}

        ids = candidates_df['fink_diaObjectId'].dropna()
        ids = ids[ids.astype(str) != 'nan']
        if max_candidates is not None:
            ids = ids.head(max_candidates)

        # Extinction lookup
        has_ext = any(f'A_{b}' in candidates_df.columns for b in 'ugriz')
        ext_lookup = {}
        if has_ext:
            for _, row in candidates_df.iterrows():
                fid = row.get('fink_diaObjectId')
                if pd.notna(fid):
                    ext = {}
                    for b in 'ugriz':
                        col = f'A_{b}'
                        if col in candidates_df.columns and pd.notna(row.get(col)):
                            ext[b] = float(row[col])
                    if ext:
                        ext_lookup[fid] = ext

        logger.info("Fitting %d candidates via Fink LSST photometry...", len(ids))

        fit_results = {}
        for i, fid in enumerate(ids):
            try:
                extinction = ext_lookup.get(fid)
                fit_results[fid] = self.fit_candidate_fink(
                    fid, extinction=extinction, use_salt=use_salt,
                )
                par_best = fit_results[fid]['parabola'].get('best')
                vil_best = fit_results[fid].get('villar', {}).get('best')
                par_s = f"para:{par_best['band']}={par_best['status']}" if par_best else "para:none"
                vil_s = f"villar:{vil_best['band']}={vil_best['status']}" if vil_best else "villar:none"
                logger.debug("[%d/%d] Fink %s — %s, %s", i + 1, len(ids), fid, par_s, vil_s)
            except Exception as e:
                logger.warning("Error fitting Fink %s: %s", fid, e)
                fit_results[fid] = {
                    'object_id': fid,
                    'broker': 'Fink-LSST',
                    'parabola': {'per_band': {}, 'best': None, 'method': 'parabola'},
                    'villar': {'per_band': {}, 'best': None, 'method': 'villar'},
                }

        n_par = sum(
            1 for r in fit_results.values()
            if r['parabola'].get('best') and r['parabola']['best']['status'] == 'ok'
        )
        n_vil = sum(
            1 for r in fit_results.values()
            if r.get('villar', {}).get('best') and r['villar']['best']['status'] == 'ok'
        )
        logger.info("Fink fitting complete: parabola %d/%d ok, villar %d/%d ok",
                    n_par, len(fit_results), n_vil, len(fit_results))
        return fit_results

    def fit_all_candidates(self, candidates_df, broker='ALeRCE-LSST',
                           id_column='object_id_ALeRCE-LSST',
                           use_salt=False, max_candidates=None):
        """Fit peak for all candidates with valid IDs.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            Candidates table from the pipeline.  If columns A_g, A_r, etc.
            are present, extinction corrections are applied per-candidate.
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

        # Build extinction lookup keyed by object ID
        has_extinction = any(f'A_{b}' in candidates_df.columns for b in 'ugriz')
        ext_lookup = {}
        if has_extinction:
            for _, row in candidates_df.iterrows():
                oid = row.get(id_column)
                if pd.notna(oid):
                    ext = {}
                    for b in 'ugriz':
                        col = f'A_{b}'
                        if col in candidates_df.columns and pd.notna(row.get(col)):
                            ext[b] = float(row[col])
                    if ext:
                        ext_lookup[oid] = ext

        logger.info("Fitting %d candidates from %s...", len(ids), broker)

        fit_results = {}
        for i, oid in enumerate(ids):
            try:
                extinction = ext_lookup.get(oid)
                fit_results[oid] = self.fit_candidate(
                    oid, broker=broker, use_salt=use_salt,
                    extinction=extinction,
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
    def enrich_with_villar(fit_results, villar_peaks):
        """Add Villar SPM peak estimates to existing fit results.

        Parameters
        ----------
        fit_results : dict
            Output from fit_all_candidates().
        villar_peaks : dict
            Output from extract_villar_peaks(): {oid: {band: result_dict}}.

        Returns
        -------
        Updated fit_results dict (modified in place).
        """
        n_enriched = 0
        for oid, band_results in villar_peaks.items():
            if oid in fit_results:
                fit_results[oid]['villar'] = band_results
                n_enriched += 1

        logger.info("Villar SPM: enriched %d/%d candidates",
                    n_enriched, len(fit_results))
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
                'peak_mag_corrected': best.get('peak_mag_corrected', np.nan) if best else np.nan,
                'A_band': best.get('A_band', np.nan) if best else np.nan,
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

            # Villar SPM columns — from direct fitting or ALeRCE features
            villar = res.get('villar')
            if villar:
                # Direct fit results have 'per_band' and 'best' keys
                if 'best' in villar and villar['best'] is not None:
                    vb = villar['best']
                    row['villar_peak_mjd'] = vb.get('peak_mjd', np.nan)
                    row['villar_peak_mag'] = vb.get('peak_mag', np.nan)
                    row['villar_peak_mag_corrected'] = vb.get('peak_mag_corrected', np.nan)
                    row['villar_chi2_dof'] = vb.get('chi2_dof', np.nan)
                    row['villar_band'] = vb.get('band', '')
                    row['villar_status'] = vb.get('status', '')
                elif 'per_band' not in villar:
                    # ALeRCE pre-computed features (dict of band → result)
                    villar_best = None
                    for b in BAND_PRIORITY:
                        if b in villar and villar[b].get('status') == 'ok':
                            villar_best = villar[b]
                            break
                    if villar_best:
                        row['villar_peak_mjd'] = villar_best.get('peak_mjd', np.nan)
                        row['villar_peak_mag'] = villar_best.get('peak_mag', np.nan)
                        row['villar_chi'] = villar_best.get('villar_chi', np.nan)
                        row['villar_band'] = villar_best.get('band', '')

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def merge_peak_fits(candidates_df, fit_results, id_column='object_id_ALeRCE-LSST'):
        """Merge headline peak estimates into the candidates DataFrame.

        Adds columns: peak_mjd, peak_mag, peak_mag_corrected, peak_band,
                      peak_mag_err, peak_fit_status.
        """
        peak_data = {}
        for oid, res in fit_results.items():
            best = res['parabola'].get('best')
            if best is not None:
                peak_data[oid] = {
                    'peak_mjd': best['peak_mjd'],
                    'peak_mag': best['peak_mag'],
                    'peak_mag_err': best.get('peak_mag_err', np.nan),
                    'peak_mag_corrected': best.get('peak_mag_corrected', np.nan),
                    'peak_band': best['band'],
                    'peak_fit_status': best['status'],
                }
            else:
                peak_data[oid] = {
                    'peak_mjd': np.nan,
                    'peak_mag': np.nan,
                    'peak_mag_err': np.nan,
                    'peak_mag_corrected': np.nan,
                    'peak_band': '',
                    'peak_fit_status': 'no_data',
                }

        peak_df = pd.DataFrame.from_dict(peak_data, orient='index')
        peak_df.index.name = id_column

        out = candidates_df.copy()
        # Drop existing peak columns to avoid conflicts
        for col in ['peak_mjd', 'peak_mag', 'peak_mag_err', 'peak_mag_corrected',
                     'peak_band', 'peak_fit_status']:
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


def plot_combined_fit(lc_df, fit_result, object_id='', figsize=(14, 7)):
    """Plot flux light curve with both parabola and Villar model overlays.

    Models and peak markers are clipped to the data time range (with a
    small margin) so that extrapolated peaks don't distort the axes.
    Flux outliers are handled via robust y-axis limits.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve with mjd, flux, flux_err, band columns.
    fit_result : dict
        Full fit result with 'parabola' and 'villar' keys.
    object_id : str
        Object label for title.
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

    parabola = fit_result.get('parabola', {})
    villar = fit_result.get('villar', {})
    par_per_band = parabola.get('per_band', {})
    vil_per_band = villar.get('per_band', {})

    bands = sorted(df['band'].dropna().unique())

    # Global data time range with margin for model curves
    data_t_min = df['mjd'].min()
    data_t_max = df['mjd'].max()
    data_span = max(data_t_max - data_t_min, 5.0)
    margin = data_span * 0.15
    plot_t_min = data_t_min - margin
    plot_t_max = data_t_max + margin

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for b in bands:
        color = BAND_COLORS.get(b, 'gray')
        mask = df['band'] == b
        sub = df[mask]

        # Separate detections from forced photometry for visual clarity
        if 'source' in sub.columns:
            det = sub[sub['source'] == 'detection']
            fp = sub[sub['source'] != 'detection']
        else:
            det = sub
            fp = pd.DataFrame()

        # Detection points (filled circles)
        if len(det) > 0:
            ax.errorbar(
                det['mjd'], det['flux'],
                yerr=det.get('flux_err'),
                fmt='o', color=color, markersize=5, alpha=0.8,
                label=f'{b} det',
            )

        # Forced photometry points (open circles, smaller)
        if len(fp) > 0:
            ax.errorbar(
                fp['mjd'], fp['flux'],
                yerr=fp.get('flux_err'),
                fmt='o', color=color, markersize=3, alpha=0.3,
                markerfacecolor='none', markeredgecolor=color,
                label=f'{b} forced',
            )

        t_grid = np.linspace(plot_t_min, plot_t_max, 400)

        # Parabola overlay (dashed)
        if b in par_per_band and par_per_band[b]['status'] not in ('insufficient_data', 'fit_failed'):
            r = par_per_band[b]
            flux_model = _inverted_parabola(t_grid, r['peak_flux'], r['peak_mjd'], r['curvature'])
            ax.plot(t_grid, flux_model, '--', color=color, alpha=0.7, linewidth=1.5,
                    label=f'{b} parabola')
            # Peak marker only if within plot range
            if plot_t_min <= r['peak_mjd'] <= plot_t_max:
                ax.plot(r['peak_mjd'], r['peak_flux'], 'D', color=color,
                        markersize=10, markeredgecolor='black', markeredgewidth=0.5,
                        zorder=5)

        # Villar overlay (solid)
        if b in vil_per_band and vil_per_band[b]['status'] not in ('insufficient_data', 'fit_failed'):
            r = vil_per_band[b]
            params = r.get('params')
            if params:
                flux_model = _villar_nJy(
                    t_grid, params['A_nJy'], params['t0_mjd'],
                    params['beta'], params['gamma'],
                    params['tau_rise'], params['tau_fall'],
                )
                ax.plot(t_grid, flux_model, '-', color=color, alpha=0.9, linewidth=2,
                        label=f'{b} Villar')
                # Peak marker only if within plot range
                if plot_t_min <= r['peak_mjd'] <= plot_t_max:
                    ax.plot(r['peak_mjd'], r['peak_flux'], '*', color=color,
                            markersize=14, markeredgecolor='black', markeredgewidth=0.5,
                            zorder=5)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Robust y-axis limits: use 1st–99th percentile of detection flux
    det_flux = df.loc[df.get('source', pd.Series(['detection'] * len(df))) == 'detection', 'flux']
    if len(det_flux) == 0:
        det_flux = df['flux']
    if len(det_flux) > 5:
        q_lo, q_hi = np.nanpercentile(det_flux, [1, 99])
        flux_range = q_hi - q_lo
        ax.set_ylim(q_lo - 0.3 * flux_range, q_hi + 0.3 * flux_range)

    ax.set_xlim(plot_t_min, plot_t_max)
    ax.set_xlabel('MJD')
    ax.set_ylabel('Flux (nJy)')

    # Build info string for title
    par_best = parabola.get('best')
    vil_best = villar.get('best')
    info_parts = []
    if par_best and np.isfinite(par_best.get('peak_mag', np.nan)):
        info_parts.append(f"Parabola: {par_best['band']}={par_best['peak_mag']:.2f} mag")
    if vil_best and np.isfinite(vil_best.get('peak_mag', np.nan)):
        info_parts.append(f"Villar: {vil_best['band']}={vil_best['peak_mag']:.2f} mag")
    info = '  |  '.join(info_parts) if info_parts else ''

    ax.set_title(f'{object_id} — Peak Fits\n{info}', fontsize=11)
    ax.legend(fontsize=7, ncol=min(len(bands) * 2, 6), loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _flux_to_mag_arrays(flux, flux_err):
    """Convert nJy flux arrays to AB magnitudes.

    Returns (mag, mag_err) with NaN for non-positive flux.
    """
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    valid = flux > 0
    mag = np.full_like(flux, np.nan)
    mag_err = np.full_like(flux, np.nan)
    mag[valid] = -2.5 * np.log10(flux[valid]) + AB_ZP_NJY
    mag_err[valid] = (2.5 / np.log(10)) * (flux_err[valid] / flux[valid])
    return mag, mag_err


def plot_mag(lc_df, fit_result, object_id='', figsize=(14, 7),
             show_forced=True, snr_floor=2.0):
    """Plot light curve and model fits in AB magnitudes.

    Only positive-flux measurements above the SNR floor are plotted
    (negative fluxes are non-detections in difference imaging).
    The y-axis is inverted (brighter = up = lower mag numbers).

    Parameters
    ----------
    lc_df : pd.DataFrame
        Light curve with mjd, flux, flux_err, band columns.
    fit_result : dict
        Full fit result with 'parabola' and/or 'villar' keys.
    object_id : str
        Object label for title.
    figsize : tuple
        Figure size.
    show_forced : bool
        Whether to show forced-photometry points.
    snr_floor : float
        Minimum flux/flux_err to plot a point (avoids noise-dominated
        measurements blowing up the magnitude axis).

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    BAND_COLORS = {
        'u': 'violet', 'g': 'green', 'r': 'red',
        'i': '#CC8800', 'z': 'purple', 'y': 'brown',
    }

    df = lc_df.copy()
    if 'psfFlux' in df.columns and 'flux' not in df.columns:
        df['flux'] = df['psfFlux']
    if 'psfFluxErr' in df.columns and 'flux_err' not in df.columns:
        df['flux_err'] = df['psfFluxErr']
    if 'band_name' in df.columns and 'band' not in df.columns:
        df['band'] = df['band_name']

    parabola = fit_result.get('parabola', {})
    villar = fit_result.get('villar', {})
    par_per_band = parabola.get('per_band', {})
    vil_per_band = villar.get('per_band', {})

    bands = sorted(df['band'].dropna().unique())

    # Time range
    data_t_min = df['mjd'].min()
    data_t_max = df['mjd'].max()
    data_span = max(data_t_max - data_t_min, 5.0)
    margin = data_span * 0.15
    plot_t_min = data_t_min - margin
    plot_t_max = data_t_max + margin
    t_grid = np.linspace(plot_t_min, plot_t_max, 500)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for b in bands:
        color = BAND_COLORS.get(b, 'gray')
        mask = df['band'] == b
        sub = df[mask].copy()

        # Only plot points with positive flux above SNR floor
        sub['snr_val'] = sub['flux'] / sub['flux_err']
        pos = sub[(sub['flux'] > 0) & (sub['snr_val'] >= snr_floor)]

        if len(pos) == 0:
            continue

        mag_vals, mag_errs = _flux_to_mag_arrays(pos['flux'].values, pos['flux_err'].values)

        # Split detections / forced
        if 'source' in pos.columns:
            det_mask = pos['source'] == 'detection'
            det = pos[det_mask]
            fp = pos[~det_mask]
        else:
            det = pos
            fp = pd.DataFrame()

        if len(det) > 0:
            det_mag, det_mag_err = _flux_to_mag_arrays(det['flux'].values, det['flux_err'].values)
            ax.errorbar(det['mjd'].values, det_mag, yerr=det_mag_err,
                        fmt='o', color=color, markersize=5, alpha=0.8,
                        label=f'{b}')

        if show_forced and len(fp) > 0:
            fp_mag, fp_mag_err = _flux_to_mag_arrays(fp['flux'].values, fp['flux_err'].values)
            ax.errorbar(fp['mjd'].values, fp_mag, yerr=fp_mag_err,
                        fmt='o', color=color, markersize=3, alpha=0.3,
                        markerfacecolor='none', markeredgecolor=color)

        # Villar model curve in magnitudes (solid)
        if b in vil_per_band and vil_per_band[b]['status'] not in ('insufficient_data', 'fit_failed'):
            r = vil_per_band[b]
            params = r.get('params')
            if params:
                flux_model = _villar_nJy(
                    t_grid, params['A_nJy'], params['t0_mjd'],
                    params['beta'], params['gamma'],
                    params['tau_rise'], params['tau_fall'],
                )
                valid = flux_model > 0
                mag_model = np.full_like(flux_model, np.nan)
                mag_model[valid] = -2.5 * np.log10(flux_model[valid]) + AB_ZP_NJY
                ax.plot(t_grid[valid], mag_model[valid], '-', color=color,
                        alpha=0.9, linewidth=2, label=f'{b} Villar')

                # Peak marker
                if (plot_t_min <= r['peak_mjd'] <= plot_t_max and
                        np.isfinite(r.get('peak_mag', np.nan))):
                    ax.plot(r['peak_mjd'], r['peak_mag'], '*', color=color,
                            markersize=14, markeredgecolor='black',
                            markeredgewidth=0.5, zorder=5)

        # Parabola model curve in magnitudes (dashed)
        if b in par_per_band and par_per_band[b]['status'] not in ('insufficient_data', 'fit_failed'):
            r = par_per_band[b]
            flux_model = _inverted_parabola(t_grid, r['peak_flux'], r['peak_mjd'], r['curvature'])
            valid = flux_model > 0
            mag_model = np.full_like(flux_model, np.nan)
            mag_model[valid] = -2.5 * np.log10(flux_model[valid]) + AB_ZP_NJY
            ax.plot(t_grid[valid], mag_model[valid], '--', color=color,
                    alpha=0.6, linewidth=1.5, label=f'{b} parab.')

            if (plot_t_min <= r['peak_mjd'] <= plot_t_max and
                    np.isfinite(r.get('peak_mag', np.nan))):
                ax.plot(r['peak_mjd'], r['peak_mag'], 'D', color=color,
                        markersize=9, markeredgecolor='black',
                        markeredgewidth=0.5, zorder=5)

    ax.invert_yaxis()  # Brighter = up
    ax.set_xlim(plot_t_min, plot_t_max)

    # Set y-axis (magnitude) limits based on data, not model curves
    # Collect all plotted detection magnitudes to set sensible range
    all_det_mags = []
    for b in bands:
        mask = df['band'] == b
        sub = df[mask]
        pos = sub[(sub['flux'] > 0) & (sub['flux'] / sub['flux_err'] >= snr_floor)]
        if 'source' in pos.columns:
            pos = pos[pos['source'] == 'detection']
        if len(pos) > 0:
            mags, _ = _flux_to_mag_arrays(pos['flux'].values, pos['flux_err'].values)
            all_det_mags.extend(mags[np.isfinite(mags)])
    if all_det_mags:
        mag_bright = min(all_det_mags) - 0.5
        mag_faint = max(all_det_mags) + 1.5
        ax.set_ylim(mag_faint, mag_bright)  # inverted: faint at bottom, bright at top

    ax.set_xlabel('MJD')
    ax.set_ylabel('AB Magnitude')

    # Title with peak info
    par_best = parabola.get('best')
    vil_best = villar.get('best')
    info_parts = []
    if vil_best and np.isfinite(vil_best.get('peak_mag', np.nan)):
        info_parts.append(f"Villar: {vil_best['band']}={vil_best['peak_mag']:.2f}")
    if par_best and np.isfinite(par_best.get('peak_mag', np.nan)):
        info_parts.append(f"Parab: {par_best['band']}={par_best['peak_mag']:.2f}")
    if villar.get('shared_t0') and np.isfinite(villar.get('shared_t0', np.nan)):
        info_parts.append(f"t0={villar['shared_t0']:.1f}")
    info = '  |  '.join(info_parts) if info_parts else ''

    ax.set_title(f'{object_id}\n{info}', fontsize=11)
    ax.legend(fontsize=8, ncol=min(len(bands) * 2, 6), loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
