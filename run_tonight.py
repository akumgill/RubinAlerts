#!/usr/bin/env python3
"""Nightly SN Ia monitoring pipeline for Magellan follow-up.

Usage:
    python run_tonight.py <MJD>
    python run_tonight.py 61100
    python run_tonight.py 61100 --min-prob 0.3 --days-back 30

Creates a night directory (e.g., nights/ut20260301/) containing:
    - candidates.csv          Summary table of all candidates
    - magellan_plan.cat       Magellan TCS catalog (RA-ordered)
    - report.pdf              Multi-page PDF with light curves and summary
    - lightcurves/            Per-candidate magnitude plots (PNG)
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.time import Time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from broker_clients.fink_client import FinkLSSTClient
from core.peak_fitting import (
    fit_parabola, fit_villar_multiband, plot_mag, clean_light_curve,
    AB_ZP_NJY, BAND_PRIORITY,
)
from core.magellan_planning import (
    compute_merit, filter_observable_targets, write_magellan_catalog,
    radec_to_sexagesimal,
)
from core.ddf_fields import DDF_FIELDS, is_in_ddf

# Multi-broker support
try:
    from supernova_monitor import SupernovaMonitor
    HAS_MONITOR = True
except ImportError:
    HAS_MONITOR = False

# Optional survey clients (for supplementary photometry)
try:
    from broker_clients.atlas_client import AtlasClient
    HAS_ATLAS = True
except ImportError:
    HAS_ATLAS = False

try:
    from broker_clients.alerce_client import AlerceClient
    HAS_ALERCE = True
except ImportError:
    HAS_ALERCE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
logger = logging.getLogger('run_tonight')


# ---------------------------------------------------------------------------
# Survey-specific photometry fetchers
# ---------------------------------------------------------------------------

def fetch_ztf_photometry(ra, dec, radius_arcsec=2.0):
    """Fetch ZTF light curve from ALeRCE by position, return in nJy flux space.

    ALeRCE returns ZTF magnitudes; we convert to nJy:
        flux_nJy = 10^((AB_ZP_NJY - mag) / 2.5)
    """
    if not HAS_ALERCE:
        return None

    try:
        alerce = AlerceClient(survey='ztf', use_db=False)
    except Exception:
        return None

    # Cone search for matching ZTF objects
    try:
        from alerce.core import Alerce
        client = Alerce()
        results = client.query_objects(
            survey='ztf',
            ra=ra, dec=dec,
            radius=radius_arcsec / 3600.0,  # degrees
            format="pandas",
            page_size=5,
        )
        if results is None or len(results) == 0:
            return None

        oid = results.iloc[0].get('oid')
        if not oid:
            return None

        # Fetch detections
        detections = client.query_detections(
            oid=oid, survey='ztf', format="pandas",
        )
        if detections is None or len(detections) == 0:
            return None

        # ZTF filter codes: 1=g, 2=r, 3=i
        ztf_band_map = {1: 'g', 2: 'r', 3: 'i'}

        # Find the magnitude columns
        mag_col = None
        for mc in ['mag_corr', 'mag', 'magpsf']:
            if mc in detections.columns:
                mag_col = mc
                break
        if mag_col is None:
            return None

        err_col = None
        for ec in ['e_mag_corr', 'e_mag', 'magerr', 'sigmapsf']:
            if ec in detections.columns:
                err_col = ec
                break

        mag = pd.to_numeric(detections[mag_col], errors='coerce').values
        mag_err = pd.to_numeric(detections.get(err_col, pd.Series(dtype=float)),
                                errors='coerce').values if err_col else np.full(len(mag), 0.05)

        valid = np.isfinite(mag) & (mag > 0) & (mag < 30)
        if not valid.any():
            return None

        # Convert mag to nJy
        flux = 10 ** ((AB_ZP_NJY - mag) / 2.5)
        # Error propagation: d(flux)/d(mag) = flux * ln(10) / 2.5
        flux_err = flux * np.log(10) / 2.5 * np.abs(mag_err)

        mjd = pd.to_numeric(detections['mjd'], errors='coerce').values
        fid = detections.get('fid', pd.Series(dtype=int))
        bands = fid.map(ztf_band_map).fillna('?').values

        df = pd.DataFrame({
            'mjd': mjd, 'flux': flux, 'flux_err': flux_err,
            'magnitude': mag, 'mag_err': mag_err,
            'band': bands, 'survey': 'ZTF', 'source': 'detection',
        })
        df = df[valid].reset_index(drop=True)

        logger.info("  ZTF (ALeRCE %s): %d detections", oid, len(df))
        return df

    except Exception as e:
        logger.debug("ZTF photometry fetch failed: %s", e)
        return None



def combine_photometry(fink_lc, ztf_lc=None, atlas_lc=None):
    """Combine light curves from multiple surveys into a single DataFrame.

    All inputs should have: mjd, flux, flux_err, band, survey columns.
    Flux should be in nJy.
    """
    frames = []

    if fink_lc is not None and len(fink_lc) > 0:
        df = fink_lc.copy()
        if 'survey' not in df.columns:
            df['survey'] = 'Rubin'
        frames.append(df)

    if ztf_lc is not None and len(ztf_lc) > 0:
        frames.append(ztf_lc)

    if atlas_lc is not None and len(atlas_lc) > 0:
        frames.append(atlas_lc)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('mjd').reset_index(drop=True)

    # Log summary
    surveys = combined['survey'].value_counts()
    logger.info("  Combined: %d points (%s)",
                len(combined),
                ', '.join(f"{s}={n}" for s, n in surveys.items()))

    return combined


def mjd_to_utdate(mjd):
    """Convert MJD to ut-style datestamp: ut20260301."""
    t = Time(mjd, format='mjd')
    return 'ut' + t.datetime.strftime('%Y%m%d')


def mjd_to_isodate(mjd):
    """Convert MJD to ISO date string: 2026-03-01."""
    t = Time(mjd, format='mjd')
    return t.datetime.strftime('%Y-%m-%d')


def fetch_fink_candidates(fink, min_sn_score=0.3, n_fetch=500):
    """Fetch SN candidates from Fink and format for the multi-broker merger."""
    logger.info("Querying Fink LSST API...")

    frames = []
    for tag in ['sn_near_galaxy_candidate', 'extragalactic_new_candidate']:
        result = fink.query_sn_candidates(tag=tag, n=n_fetch)
        if result is not None and len(result) > 0:
            result['fink_tag'] = tag
            frames.append(result)

    if not frames:
        logger.warning("No candidates from Fink")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

    # Parse scores
    raw['sn_score'] = pd.to_numeric(
        raw.get('f:clf_snnSnVsOthers_score', pd.Series(dtype=float)),
        errors='coerce',
    )
    raw['early_ia_score'] = pd.to_numeric(
        raw.get('f:clf_earlySNIa_score', pd.Series(dtype=float)),
        errors='coerce',
    )

    # Filter by SN score
    good = raw[raw['sn_score'] >= min_sn_score].copy()

    # Deduplicate by diaObjectId
    good['diaObjectId'] = good['r:diaObjectId'].astype(str)
    good = good.sort_values('sn_score', ascending=False)
    good = good.drop_duplicates(subset='diaObjectId', keep='first')

    # Normalize columns for the aggregator
    good['object_id'] = good['diaObjectId']
    good['ra'] = pd.to_numeric(good.get('r:ra', pd.Series(dtype=float)), errors='coerce')
    good['dec'] = pd.to_numeric(good.get('r:dec', pd.Series(dtype=float)), errors='coerce')
    good['sn_ia_prob'] = good['sn_score']  # aggregator expects this column
    good['broker'] = 'Fink'

    # Assign DDF fields
    good['ddf_field'] = good.apply(
        lambda r: is_in_ddf(r['ra'], r['dec']) if pd.notna(r['ra']) else None,
        axis=1,
    )

    logger.info("Fink: %d candidates (score >= %.2f)", len(good), min_sn_score)
    return good


def fetch_all_broker_candidates(fink, min_prob=0.3, days_back=30, n_fetch=500,
                                fink_only=False):
    """Query all brokers for SN candidates, merge, deduplicate, screen variables.

    Brokers queried:
      - Fink LSST (always)
      - ALeRCE-ZTF (if available)
      - ALeRCE-LSST (if available)
      - ANTARES (if available)

    Returns a merged, deduplicated DataFrame with columns:
      diaObjectId, ra, dec, ddf_field, sn_score, brokers_detected,
      num_brokers, mean_ia_prob, known_variable, ...
    """
    # --- Always query Fink ---
    fink_df = fetch_fink_candidates(fink, min_sn_score=min_prob, n_fetch=n_fetch)

    if fink_only or not HAS_MONITOR:
        if not HAS_MONITOR and not fink_only:
            logger.warning("SupernovaMonitor not available; using Fink only")
        # Fink-only path: just assign fields and return
        if len(fink_df) == 0:
            return pd.DataFrame()
        fink_df['brokers_detected'] = 'Fink'
        fink_df['num_brokers'] = 1
        fink_df['mean_ia_prob'] = fink_df['sn_score']
        fink_df['known_variable'] = False
        return fink_df

    # --- Query other brokers via SupernovaMonitor ---
    logger.info("Querying ANTARES + ALeRCE brokers...")
    try:
        monitor = SupernovaMonitor(cache_dir='./cache/data')
        other_brokers = monitor.query_all_brokers(
            class_name='SNIa',
            min_probability=min_prob,
            days_back=days_back,
            limit=n_fetch,
            ddf_fields=DDF_FIELDS,
        )
    except Exception as e:
        logger.warning("Multi-broker query failed: %s. Using Fink only.", e)
        if len(fink_df) == 0:
            return pd.DataFrame()
        fink_df['brokers_detected'] = 'Fink'
        fink_df['num_brokers'] = 1
        fink_df['mean_ia_prob'] = fink_df['sn_score']
        fink_df['known_variable'] = False
        return fink_df

    # Log per-broker counts
    for broker_name, bdf in other_brokers.items():
        n = len(bdf) if bdf is not None else 0
        logger.info("  %s: %d candidates", broker_name, n)

    # --- Add Fink to the broker dict ---
    if len(fink_df) > 0:
        other_brokers['Fink'] = fink_df

    # --- Merge and deduplicate across all brokers ---
    aggregator = monitor.aggregator
    merged = aggregator.merge_alerts(other_brokers)

    if len(merged) == 0:
        logger.warning("No candidates after merge")
        return pd.DataFrame()

    logger.info("Merged across brokers: %d unique candidates", len(merged))

    # --- Screen against known variable catalogs ---
    try:
        screened = monitor.variable_screener.screen_candidates(merged)
        n_var = screened['known_variable'].sum() if 'known_variable' in screened.columns else 0
        if n_var > 0:
            logger.info("Flagged %d known variables — removing", n_var)
            screened = screened[~screened['known_variable']].copy()
    except Exception as e:
        logger.warning("Variable screening failed: %s", e)
        screened = merged

    # --- Filter by P(Ia) ---
    if 'mean_ia_prob' in screened.columns:
        before = len(screened)
        screened = screened[screened['mean_ia_prob'] >= min_prob].copy()
        logger.info("After P(Ia) >= %.2f: %d (dropped %d)",
                    min_prob, len(screened), before - len(screened))

    # --- Normalize output columns for downstream compatibility ---
    # Need: diaObjectId, ra, dec, ddf_field, sn_score
    if 'diaObjectId' not in screened.columns:
        # Use rubin_dia_object_id if available, else unique_id or object_id
        if 'rubin_dia_object_id' in screened.columns:
            screened['diaObjectId'] = screened['rubin_dia_object_id'].fillna(
                screened.get('unique_id', screened.get('object_id', ''))
            )
        elif 'unique_id' in screened.columns:
            screened['diaObjectId'] = screened['unique_id']
        elif 'object_id' in screened.columns:
            screened['diaObjectId'] = screened['object_id']
        else:
            screened['diaObjectId'] = [f"obj_{i}" for i in range(len(screened))]

    if 'sn_score' not in screened.columns:
        screened['sn_score'] = screened.get('mean_ia_prob', np.nan)

    if 'ddf_field' not in screened.columns:
        screened['ddf_field'] = screened.apply(
            lambda r: is_in_ddf(r['ra'], r['dec']) if pd.notna(r.get('ra')) else None,
            axis=1,
        )

    logger.info("Final candidates: %d from %s",
                len(screened),
                ', '.join(sorted(set(screened.get('brokers_detected', pd.Series(['Fink'])).dropna()))))

    return screened


ATLAS_BRIGHT_MAG_CUT = 20.0  # Only fetch ATLAS for candidates brighter than this


def _atlas_filter_to_nJy(by_filter):
    """Convert ATLAS per-filter photometry dict to unified nJy DataFrame.

    Parameters
    ----------
    by_filter : dict with keys 'c' and/or 'o', values are ATLAS DataFrames
                with columns MJD, uJy, duJy, etc.

    Returns
    -------
    DataFrame in standard photometry format (mjd, flux, flux_err, band, ...)
    or None if no data.
    """
    atlas_band_names = {'c': 'ATLAS-c', 'o': 'ATLAS-o'}
    frames = []
    for filt, fdf in by_filter.items():
        if fdf is None or len(fdf) == 0:
            continue
        uJy = fdf['uJy'].values.astype(float)
        duJy = fdf['duJy'].values.astype(float)
        mjd_vals = fdf['MJD'].values.astype(float)

        flux_nJy = uJy * 1000.0
        flux_err_nJy = duJy * 1000.0

        valid = uJy > 0
        mag = np.full(len(fdf), np.nan)
        mag_err = np.full(len(fdf), np.nan)
        mag[valid] = -2.5 * np.log10(uJy[valid]) + 23.9
        mag_err[valid] = 1.0857 * duJy[valid] / uJy[valid]

        frames.append(pd.DataFrame({
            'mjd': mjd_vals, 'flux': flux_nJy, 'flux_err': flux_err_nJy,
            'magnitude': mag, 'mag_err': mag_err,
            'band': atlas_band_names.get(filt, filt),
            'survey': 'ATLAS', 'source': 'forced_phot',
        }))

    if frames:
        return pd.concat(frames, ignore_index=True).sort_values('mjd').reset_index(drop=True)
    return None


def fetch_and_fit(fink, candidates_df, mjd_now, fetch_ztf=True, fetch_atlas=True):
    """Fetch light curves from all surveys and run peak fitting for each candidate.

    Two-pass approach:
      Pass 1: Fetch Fink/Rubin photometry for all candidates, identify bright ones.
      Batch:  Submit bright candidates (any Rubin detection < 20 mag) to ATLAS
              as a single batch, and fetch ZTF per-candidate.
      Pass 2: Combine all photometry and run fits.
    """
    dia_ids = candidates_df['diaObjectId'].unique()
    logger.info("Fitting %d candidates...", len(dia_ids))

    # Lookup RA/Dec for each candidate
    coord_lookup = {}
    for _, row in candidates_df.iterrows():
        coord_lookup[row['diaObjectId']] = (row['ra'], row['dec'])

    # MJD lower bound for ATLAS queries (avoid fetching years of old data)
    atlas_mjd_min = mjd_now - 180  # 6 months back

    # ---- Pass 1: Fetch Fink photometry and identify bright candidates ----
    fink_data = {}  # did -> DataFrame
    bright_for_atlas = []  # (did, ra, dec) for candidates brighter than cut
    for i, did in enumerate(dia_ids):
        logger.info("[%d/%d] Fink: %s", i + 1, len(dia_ids), did)
        fink_lc = fink.get_light_curve(str(did), include_forced=True)
        if fink_lc is None or len(fink_lc) == 0:
            logger.warning("  No Fink light curve")
            continue
        fink_data[did] = fink_lc

        # Check if any Rubin detection is brighter than the ATLAS cut
        if fetch_atlas and 'magnitude' in fink_lc.columns:
            mags = pd.to_numeric(fink_lc['magnitude'], errors='coerce')
            brightest = mags.dropna().min()
            if np.isfinite(brightest) and brightest < ATLAS_BRIGHT_MAG_CUT:
                ra, dec = coord_lookup.get(did, (np.nan, np.nan))
                if np.isfinite(ra) and np.isfinite(dec):
                    bright_for_atlas.append((str(did), ra, dec))

    logger.info("Fink photometry: %d/%d candidates have data", len(fink_data), len(dia_ids))

    # ---- Batch ATLAS for bright candidates ----
    atlas_data = {}  # did -> DataFrame (nJy format)
    if fetch_atlas and bright_for_atlas and HAS_ATLAS:
        logger.info("ATLAS: %d candidates brighter than %.1f mag — submitting batch",
                     len(bright_for_atlas), ATLAS_BRIGHT_MAG_CUT)
        try:
            atlas_client = AtlasClient()
            batch_phot = atlas_client.fetch_batch_photometry(
                bright_for_atlas, mjd_min=atlas_mjd_min)
            for oid, by_filter in batch_phot.items():
                atlas_df = _atlas_filter_to_nJy(by_filter)
                if atlas_df is not None and len(atlas_df) > 0:
                    atlas_data[oid] = atlas_df
            logger.info("ATLAS: %d/%d candidates returned photometry",
                         len(atlas_data), len(bright_for_atlas))
        except Exception as e:
            logger.warning("ATLAS batch fetch failed: %s", e)
    elif fetch_atlas and not bright_for_atlas:
        logger.info("ATLAS: no candidates brighter than %.1f mag — skipping",
                     ATLAS_BRIGHT_MAG_CUT)

    # ---- Pass 2: Fetch ZTF, combine, and fit ----
    results = {}
    for i, did in enumerate(fink_data):
        logger.info("[%d/%d] Fitting %s", i + 1, len(fink_data), did)
        ra, dec = coord_lookup.get(did, (np.nan, np.nan))
        fink_lc = fink_data[did]

        # ZTF photometry (per-candidate, fast API query)
        ztf_lc = None
        if fetch_ztf and np.isfinite(ra) and np.isfinite(dec):
            try:
                ztf_lc = fetch_ztf_photometry(ra, dec)
            except Exception as e:
                logger.debug("  ZTF fetch error: %s", e)

        # ATLAS photometry (from batch results)
        atlas_lc = atlas_data.get(str(did))

        # --- Combine all photometry ---
        combined = combine_photometry(fink_lc, ztf_lc, atlas_lc)
        if combined is None:
            combined = fink_lc  # fallback to Fink-only

        lc_clean = clean_light_curve(combined)
        if len(lc_clean) < 5:
            logger.warning("  Too few points after cleaning (%d)", len(lc_clean))
            continue

        band_counts = lc_clean.groupby('band').size()
        logger.info("  %d clean pts: %s", len(lc_clean),
                    ', '.join(f"{b}={n}" for b, n in band_counts.items()))

        # --- Run both fits on combined data ---
        par = fit_parabola(combined)
        vil = fit_villar_multiband(combined)

        # Extract peak info
        par_best = par.get('best')
        vil_best = vil.get('best')

        # Pick the better fit for the headline peak
        best = vil_best if vil_best and vil_best.get('status') == 'ok' else par_best
        if best is None or best.get('status') != 'ok':
            best = par_best  # fallback

        peak_mag = best.get('peak_mag', np.nan) if best else np.nan
        peak_mjd = best.get('peak_mjd', np.nan) if best else np.nan
        peak_band = best.get('band', '') if best else ''
        fit_method = 'villar_mb' if (vil_best and vil_best.get('status') == 'ok') else 'parabola'

        delta_t = mjd_now - peak_mjd if np.isfinite(peak_mjd) else np.nan

        # Track survey coverage
        surveys_present = combined['survey'].unique().tolist() if 'survey' in combined.columns else ['Rubin']
        n_ztf = len(ztf_lc) if ztf_lc is not None else 0
        n_atlas = len(atlas_lc) if atlas_lc is not None else 0

        results[did] = {
            'diaObjectId': did,
            'parabola': par,
            'villar': vil,
            'light_curve': combined,
            'light_curve_clean': lc_clean,
            'peak_mag': peak_mag,
            'peak_mjd': peak_mjd,
            'peak_band': peak_band,
            'delta_t': delta_t,
            'fit_method': fit_method,
            'n_points': len(lc_clean),
            'n_bands': lc_clean['band'].nunique(),
            'surveys': surveys_present,
            'n_ztf': n_ztf,
            'n_atlas': n_atlas,
        }

        # Log summary
        survey_str = '+'.join(surveys_present)
        if np.isfinite(peak_mag):
            logger.info("  Peak: %s=%s=%.2f at MJD %.1f (dt=%.1fd) [%s]",
                        peak_band, fit_method, peak_mag, peak_mjd, delta_t, survey_str)
        else:
            logger.info("  No good peak fit [%s]", survey_str)

    return results


def build_summary_table(candidates_df, fit_results, mjd_now):
    """Build a merged summary DataFrame with merit scores."""
    rows = []
    for _, cand in candidates_df.iterrows():
        did = cand['diaObjectId']
        fit = fit_results.get(did)
        if fit is None:
            continue

        peak_mag = fit['peak_mag']
        delta_t = fit['delta_t']

        # Merit score
        merit = compute_merit(delta_t, peak_mag) if (
            np.isfinite(delta_t) and np.isfinite(peak_mag)
        ) else np.nan

        rows.append({
            'diaObjectId': did,
            'ra': cand['ra'],
            'dec': cand['dec'],
            'ddf_field': cand.get('ddf_field', ''),
            'sn_score': cand.get('sn_score', np.nan),
            'early_ia_score': cand.get('early_ia_score', np.nan),
            'brokers_detected': cand.get('brokers_detected', 'Fink'),
            'num_brokers': cand.get('num_brokers', 1),
            'mean_ia_prob': cand.get('mean_ia_prob', np.nan),
            'peak_mag': peak_mag,
            'peak_mjd': fit['peak_mjd'],
            'peak_band': fit['peak_band'],
            'delta_t': delta_t,
            'fit_method': fit['fit_method'],
            'n_points': fit['n_points'],
            'n_bands': fit['n_bands'],
            'surveys': '+'.join(fit.get('surveys', ['Rubin'])),
            'n_ztf': fit.get('n_ztf', 0),
            'n_atlas': fit.get('n_atlas', 0),
            'merit': merit,
            'object_id': did,  # alias for magellan_planning
        })

    summary = pd.DataFrame(rows)
    if len(summary) > 0:
        summary = summary.sort_values('merit', ascending=False, na_position='last')
    return summary


def generate_light_curve_plots(fit_results, lc_dir, summary_df):
    """Generate per-candidate magnitude plots, return list of figure paths."""
    os.makedirs(lc_dir, exist_ok=True)
    plot_paths = {}

    for did, fit in fit_results.items():
        lc_clean = fit['light_curve_clean']
        try:
            fig = plot_mag(
                lc_clean, fit,
                object_id=f'{did}',
                figsize=(12, 6),
            )
            fname = os.path.join(lc_dir, f'{did[-12:]}.png')
            fig.savefig(fname, dpi=120, bbox_inches='tight')
            plt.close(fig)
            plot_paths[did] = fname
        except Exception as e:
            logger.warning("Failed to plot %s: %s", did, e)

    logger.info("Generated %d light curve plots in %s", len(plot_paths), lc_dir)
    return plot_paths


def generate_pdf_report(summary_df, fit_results, plot_paths,
                        pdf_path, mjd_now, obs_date):
    """Generate multi-page PDF report with summary and light curves."""
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_path) as pdf:
        # --- Page 1: Title + Summary Table ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        title_lines = [
            f'SN Ia Monitoring Report',
            f'MJD {mjd_now:.1f}  |  {obs_date}  |  {mjd_to_utdate(mjd_now)}',
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} UT',
            f'{len(summary_df)} candidates with peak fits',
        ]
        ax.text(0.5, 0.95, '\n'.join(title_lines),
                ha='center', va='top', fontsize=14,
                fontfamily='monospace', transform=ax.transAxes)

        # Summary table
        if len(summary_df) > 0:
            table_df = summary_df.head(30).copy()
            table_df['RA_s'] = table_df.apply(
                lambda r: radec_to_sexagesimal(r['ra'], r['dec'])[0], axis=1)
            table_df['Dec_s'] = table_df.apply(
                lambda r: radec_to_sexagesimal(r['ra'], r['dec'])[1], axis=1)

            display_cols = ['diaObjectId', 'ddf_field', 'RA_s', 'Dec_s',
                           'peak_mag', 'peak_band', 'delta_t', 'merit',
                           'brokers_detected', 'num_brokers', 'fit_method',
                           'surveys']
            display_df = table_df[
                [c for c in display_cols if c in table_df.columns]
            ].copy()

            # Format numbers
            for col in ['peak_mag', 'merit', 'sn_score']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f'{x:.2f}' if pd.notna(x) and np.isfinite(x) else '--')
            if 'delta_t' in display_df.columns:
                display_df['delta_t'] = display_df['delta_t'].apply(
                    lambda x: f'{x:+.1f}d' if pd.notna(x) and np.isfinite(x) else '--')

            # Shorten diaObjectId for display
            display_df['diaObjectId'] = display_df['diaObjectId'].str[-10:]

            tbl = ax.table(
                cellText=display_df.values,
                colLabels=display_df.columns,
                loc='center',
                cellLoc='center',
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.auto_set_column_width(range(len(display_df.columns)))
            tbl.scale(1.0, 1.3)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # --- Page 2: Merit vs peak_mag scatter ---
        if len(summary_df) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

            # Merit vs magnitude
            ax = axes[0]
            valid = summary_df[summary_df['merit'].notna()]
            ax.scatter(valid['peak_mag'], valid['merit'], c='steelblue', s=20, alpha=0.7)
            ax.set_xlabel('Peak Magnitude (AB)')
            ax.set_ylabel('Merit Score')
            ax.set_title('Merit vs Peak Brightness')
            ax.grid(True, alpha=0.3)

            # Merit vs delta_t
            ax = axes[1]
            ax.scatter(valid['delta_t'], valid['merit'], c='firebrick', s=20, alpha=0.7)
            ax.set_xlabel('Days Since Peak')
            ax.set_ylabel('Merit Score')
            ax.set_title('Merit vs Time from Peak')
            ax.grid(True, alpha=0.3)

            # Sky distribution
            ax = axes[2]
            ax.scatter(summary_df['ra'], summary_df['dec'],
                      c=summary_df['merit'].fillna(0), cmap='YlOrRd',
                      s=30, alpha=0.7, edgecolors='gray', linewidths=0.3)
            for f in DDF_FIELDS:
                ax.annotate(f['name'], (f['ra'], f['dec']),
                          fontsize=7, ha='center', alpha=0.5)
            ax.set_xlabel('RA (deg)')
            ax.set_ylabel('Dec (deg)')
            ax.set_title('Sky Distribution')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # --- Page 3: Delta-t vs Peak Magnitude, color-coded by merit ---
        if len(summary_df) > 0:
            valid = summary_df.dropna(subset=['peak_mag', 'delta_t'])
            if len(valid) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                merit_vals = valid['merit'].fillna(0).values
                sc = ax.scatter(valid['delta_t'], valid['peak_mag'],
                                c=merit_vals, cmap='YlOrRd', s=40, alpha=0.8,
                                edgecolors='gray', linewidths=0.3,
                                vmin=0, vmax=max(merit_vals.max(), 0.01))
                cbar = plt.colorbar(sc, ax=ax, label='Merit Score')

                # Annotate high-merit targets
                high_merit = valid[valid['merit'] > 0.3]
                for _, row in high_merit.iterrows():
                    label = str(row['diaObjectId'])[-6:]  # last 6 digits
                    ax.annotate(label, (row['delta_t'], row['peak_mag']),
                                fontsize=6, alpha=0.7,
                                xytext=(4, 4), textcoords='offset points')

                ax.set_xlabel('Days Since Peak (negative = pre-peak)')
                ax.set_ylabel('Peak Magnitude (AB)')
                ax.invert_yaxis()  # brighter at top
                ax.set_title(f'Discovery Space — {len(valid)} candidates with fits')
                ax.axvline(0, color='gray', linestyle='--', alpha=0.4, label='Peak')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # --- Remaining pages: light curve plots, 4 per page ---
        # Sort by merit (best first)
        ordered = summary_df.sort_values('merit', ascending=False, na_position='last')
        plot_dids = [did for did in ordered['diaObjectId'] if did in plot_paths]

        for page_start in range(0, len(plot_dids), 4):
            page_dids = plot_dids[page_start:page_start + 4]
            n = len(page_dids)
            fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n))
            if n == 1:
                axes = [axes]

            for ax, did in zip(axes, page_dids):
                img = plt.imread(plot_paths[did])
                ax.imshow(img)
                ax.axis('off')

                # Add merit annotation
                row = summary_df[summary_df['diaObjectId'] == did]
                if len(row) > 0:
                    r = row.iloc[0]
                    info = f"Merit={r['merit']:.3f}" if pd.notna(r['merit']) else ""
                    if pd.notna(r.get('ddf_field')):
                        info += f"  DDF={r['ddf_field']}"
                    ax.set_title(info, fontsize=9, loc='right')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    logger.info("PDF report: %s (%d pages)", pdf_path,
                2 + (len(plot_dids) + 3) // 4)


def generate_observing_schedule(plan_df, mjd_now, obs_date, output_path):
    """Generate a human-readable observing schedule ordered by RA.

    Assumes 30 minutes per observation. Lists targets in RA order
    with coordinates, magnitude, merit, and estimated UT time.
    """
    if len(plan_df) == 0:
        return

    df = plan_df.sort_values('ra').reset_index(drop=True)

    lines = []
    lines.append(f'# Magellan Observing Schedule — {obs_date} UT')
    lines.append(f'# MJD {mjd_now:.1f} | {len(df)} targets | ~30 min each')
    lines.append(f'# Sorted by RA for efficient slewing')
    lines.append('#')
    lines.append(f'# {"#":>3s}  {"Object":20s}  {"RA":>11s}  {"Dec":>10s}  '
                 f'{"Band":>4s}  {"PkMag":>6s}  {"dt":>6s}  {"Merit":>6s}  '
                 f'{"DDF":>8s}  {"Brokers":>20s}  {"Phot":>16s}')
    lines.append(f'# {"---":>3s}  {"----":20s}  {"--":>11s}  {"---":>10s}  '
                 f'{"----":>4s}  {"-----":>6s}  {"--":>6s}  {"-----":>6s}  '
                 f'{"---":>8s}  {"-------":>20s}  {"----":>16s}')

    for i, (_, row) in enumerate(df.iterrows()):
        ra_s, dec_s = radec_to_sexagesimal(row['ra'], row['dec'])

        pmag = f"{row['peak_mag']:.1f}" if np.isfinite(row.get('peak_mag', np.nan)) else '--'
        dt = f"{row['delta_t']:+.0f}d" if np.isfinite(row.get('delta_t', np.nan)) else '--'
        merit = f"{row['merit']:.3f}" if np.isfinite(row.get('merit', np.nan)) else '--'
        ddf = row.get('ddf_field', '') or ''
        band = row.get('peak_band', '') or ''
        brokers = row.get('brokers_detected', 'Fink') or 'Fink'
        surveys = row.get('surveys', 'Rubin') or 'Rubin'
        did = str(row['diaObjectId'])[-12:]

        lines.append(f'  {i+1:3d}  {did:20s}  {ra_s:>11s}  {dec_s:>10s}  '
                     f'{band:>4s}  {pmag:>6s}  {dt:>6s}  {merit:>6s}  '
                     f'{ddf:>8s}  {brokers:>20s}  {surveys:>16s}')

    lines.append('#')
    lines.append(f'# Total observing time: ~{len(df) * 0.5:.1f} hours '
                 f'({len(df)} targets x 30 min)')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info("Observing schedule: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description='SN Ia nightly monitoring pipeline for Magellan follow-up',
        epilog='Example: python run_tonight.py 61100',
    )
    parser.add_argument('mjd', type=float, help='MJD of the observing night')
    parser.add_argument('--min-prob', type=float, default=0.3,
                        help='Minimum SN score (default: 0.3)')
    parser.add_argument('--days-back', type=int, default=30,
                        help='Days of history to query (default: 30)')
    parser.add_argument('--max-candidates', type=int, default=200,
                        help='Max candidates to fetch (default: 200)')
    parser.add_argument('--instrument', type=str, default='LDSS3C',
                        help='Magellan instrument (default: LDSS3C)')
    parser.add_argument('--output-dir', type=str, default='nights',
                        help='Base output directory (default: nights)')
    parser.add_argument('--max-airmass', type=float, default=2.0,
                        help='Maximum airmass for observability (default: 2.0)')
    parser.add_argument('--no-observability', action='store_true',
                        help='Skip observability filtering')
    parser.add_argument('--no-ztf', action='store_true',
                        help='Skip ZTF photometry from ALeRCE')
    parser.add_argument('--no-atlas', action='store_true',
                        help='Skip ATLAS forced photometry')
    parser.add_argument('--fink-only', action='store_true',
                        help='Only query Fink (skip ANTARES/ALeRCE brokers)')

    args = parser.parse_args()
    mjd_now = args.mjd

    # Compute date strings
    ut_stamp = mjd_to_utdate(mjd_now)
    obs_date = mjd_to_isodate(mjd_now)

    logger.info("=" * 70)
    logger.info("SN Ia Nightly Pipeline")
    logger.info("MJD: %.1f  |  Date: %s  |  Stamp: %s", mjd_now, obs_date, ut_stamp)
    logger.info("=" * 70)

    # Create night directory
    night_dir = os.path.join(args.output_dir, ut_stamp)
    lc_dir = os.path.join(night_dir, 'lightcurves')
    os.makedirs(lc_dir, exist_ok=True)
    logger.info("Output directory: %s", night_dir)

    # --- Step 1: Connect to Fink ---
    fink = FinkLSSTClient()
    if not fink.available:
        logger.error("Fink LSST API not reachable")
        sys.exit(1)
    logger.info("Fink LSST API: connected")

    # --- Step 2: Query all brokers, merge, deduplicate, screen variables ---
    if args.fink_only:
        logger.info("Mode: Fink only (--fink-only)")
    else:
        logger.info("Mode: All brokers (Fink + ANTARES + ALeRCE-ZTF + ALeRCE-LSST)")
    candidates = fetch_all_broker_candidates(
        fink,
        min_prob=args.min_prob,
        days_back=args.days_back,
        n_fetch=args.max_candidates,
        fink_only=args.fink_only,
    )
    if len(candidates) == 0:
        logger.error("No candidates found")
        sys.exit(1)

    # --- Step 3: Fetch light curves and fit peaks ---
    # ZTF: per-candidate ALeRCE queries; ATLAS: batch for bright candidates only
    do_ztf = not args.no_ztf
    do_atlas = not args.no_atlas
    if do_ztf:
        logger.info("ZTF photometry: %s", "enabled (via ALeRCE)" if HAS_ALERCE else "SKIPPED (alerce not installed)")
    if do_atlas:
        logger.info("ATLAS photometry: %s (batch, bright < %.1f mag only)",
                     "enabled" if HAS_ATLAS else "SKIPPED (atlas_client not available)",
                     ATLAS_BRIGHT_MAG_CUT)
    fit_results = fetch_and_fit(fink, candidates, mjd_now,
                                fetch_ztf=do_ztf and HAS_ALERCE,
                                fetch_atlas=do_atlas and HAS_ATLAS)
    if not fit_results:
        logger.error("No successful fits")
        sys.exit(1)
    logger.info("Successful fits: %d / %d", len(fit_results), len(candidates))

    # --- Step 4: Build summary table with merit scores ---
    summary = build_summary_table(candidates, fit_results, mjd_now)
    logger.info("Summary table: %d rows", len(summary))

    if len(summary) == 0:
        logger.error("Empty summary table")
        sys.exit(1)

    # --- Step 5: Observability filter ---
    if not args.no_observability:
        logger.info("Filtering for observability from Las Campanas on %s...", obs_date)
        try:
            plan = filter_observable_targets(
                summary, obs_date,
                max_airmass=args.max_airmass,
                min_hours_up=0.5,
            )
        except Exception as e:
            logger.warning("Observability calculation failed: %s. Using all targets.", e)
            plan = summary.copy()
    else:
        plan = summary.copy()

    # Sort by RA for observing efficiency
    plan = plan.sort_values('ra').reset_index(drop=True)
    logger.info("Observing plan: %d targets (RA-ordered)", len(plan))

    # --- Step 6: Generate light curve plots ---
    plot_paths = generate_light_curve_plots(fit_results, lc_dir, summary)

    # --- Step 7: Save outputs ---
    # Summary CSV
    csv_path = os.path.join(night_dir, 'candidates.csv')
    summary.to_csv(csv_path, index=False)
    logger.info("Candidates CSV: %s", csv_path)

    # Magellan catalog
    cat_path = os.path.join(night_dir, 'magellan_plan.cat')
    write_magellan_catalog(
        plan, cat_path,
        instrument=args.instrument,
        obs_date=obs_date,
    )

    # Human-readable schedule
    sched_path = os.path.join(night_dir, 'observing_schedule.txt')
    generate_observing_schedule(plan, mjd_now, obs_date, sched_path)

    # PDF report
    pdf_path = os.path.join(night_dir, 'report.pdf')
    generate_pdf_report(summary, fit_results, plot_paths,
                        pdf_path, mjd_now, obs_date)

    # --- Done ---
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Night directory: %s", night_dir)
    logger.info("  candidates.csv         %d candidates", len(summary))
    logger.info("  magellan_plan.cat      %d targets (RA-ordered)", len(plan))
    logger.info("  observing_schedule.txt readable schedule")
    logger.info("  report.pdf             summary + light curves")
    logger.info("  lightcurves/           %d plots", len(plot_paths))

    # Survey coverage summary
    if 'surveys' in summary.columns:
        n_with_ztf = (summary['n_ztf'] > 0).sum() if 'n_ztf' in summary.columns else 0
        n_with_atlas = (summary['n_atlas'] > 0).sum() if 'n_atlas' in summary.columns else 0
        logger.info("  Survey coverage: %d with ZTF, %d with ATLAS", n_with_ztf, n_with_atlas)

    # Print top 5 by merit
    if len(plan) > 0:
        top = plan.sort_values('merit', ascending=False).head(5)
        logger.info("\nTop 5 by merit:")
        for _, r in top.iterrows():
            ra_s, dec_s = radec_to_sexagesimal(r['ra'], r['dec'])
            logger.info("  %s  %s %s  mag=%.1f  dt=%+.0fd  merit=%.3f  %s",
                        str(r['diaObjectId'])[-12:], ra_s, dec_s,
                        r['peak_mag'] if np.isfinite(r['peak_mag']) else 0,
                        r['delta_t'] if np.isfinite(r['delta_t']) else 0,
                        r['merit'] if np.isfinite(r['merit']) else 0,
                        r.get('ddf_field', ''))

    # Print RA-ordered schedule summary
    if len(plan) > 0:
        logger.info("\nRA-ordered schedule (%d targets, ~%.1f hours):",
                    len(plan), len(plan) * 0.5)
        for i, (_, r) in enumerate(plan.iterrows()):
            ra_s, _ = radec_to_sexagesimal(r['ra'], r['dec'])
            logger.info("  %2d. %s  RA=%s  mag=%.1f  merit=%.3f",
                        i + 1, str(r['diaObjectId'])[-12:], ra_s,
                        r['peak_mag'] if np.isfinite(r['peak_mag']) else 0,
                        r['merit'] if np.isfinite(r['merit']) else 0)


if __name__ == '__main__':
    main()
