#!/usr/bin/env python3
"""Nightly SN Ia monitoring pipeline for Magellan follow-up.

Usage:
    python run_tonight.py <MJD>
    python run_tonight.py 61100
    python run_tonight.py 61100 --min-prob 0.3 --days-back 30

Creates a night directory (e.g., nights/ut20260301/) containing:
    - candidates.csv          Summary table of all candidates
    - magellan_plan.cat       Magellan TCS catalog (RA-ordered)
    - report_{ut_stamp}.pdf   Multi-page PDF with light curves and summary
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
    fit_parabola, fit_villar_multiband, fit_salt, plot_mag, clean_light_curve,
    AB_ZP_NJY, BAND_PRIORITY, HAS_SNCOSMO,
)
from core.magellan_planning import (
    compute_merit, compute_merit_breakdown, filter_observable_targets,
    write_magellan_catalog, radec_to_sexagesimal, prioritize_targets,
    optimize_observing_sequence,
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

try:
    from host_galaxy.morphology_filter import MorphologyFilter
    HAS_MORPHOLOGY = True
except ImportError:
    HAS_MORPHOLOGY = False

# NED redshift queries
try:
    from utils.ned_query import query_ned_batch, query_ned_redshift
    HAS_NED = True
except ImportError:
    HAS_NED = False

# TNS (Transient Name Server) cross-matching
try:
    from broker_clients.tns_client import TNSClient
    HAS_TNS = True
except ImportError:
    HAS_TNS = False

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


def fetch_ztf_photometry_batch(positions, radius_arcsec=2.0):
    """Batch fetch ZTF light curves from ALeRCE via direct DB access.

    Much faster than per-candidate REST API queries. Uses spatial cross-match
    to find ZTF OIDs, then batch-fetches detections.

    Parameters
    ----------
    positions : list of (id, ra, dec) tuples
        Candidate positions to fetch ZTF photometry for.
    radius_arcsec : float
        Cross-match radius in arcseconds.

    Returns
    -------
    dict of id -> DataFrame with ZTF photometry in nJy flux space
    """
    if not HAS_ALERCE:
        return {}

    try:
        from broker_clients.alerce_db_client import AlerceDBClient
        db = AlerceDBClient()
        if not db.available:
            logger.info("ZTF batch: DB client not available, skipping")
            return {}

        db.connect()

        # Phase 1: Cross-match positions to ZTF OIDs
        logger.info("ZTF batch: cross-matching %d positions...", len(positions))
        id_to_oid = db.crossmatch_positions(positions, radius_arcsec=radius_arcsec)

        if not id_to_oid:
            logger.info("ZTF batch: no cross-matches found")
            return {}

        # Phase 2: Batch fetch detections for all matched OIDs
        unique_oids = list(set(id_to_oid.values()))
        logger.info("ZTF batch: fetching detections for %d unique ZTF objects...",
                   len(unique_oids))
        detections = db.query_detections(unique_oids)

        if detections is None or len(detections) == 0:
            logger.info("ZTF batch: no detections returned")
            return {}

        # Phase 3: Convert to nJy flux and group by original ID
        # ZTF filter codes: 1=g, 2=r, 3=i
        ztf_band_map = {1: 'g', 2: 'r', 3: 'i'}

        results = {}
        oid_to_ids = {}  # reverse mapping: oid -> list of original IDs
        for pid, oid in id_to_oid.items():
            oid_to_ids.setdefault(oid, []).append(pid)

        for oid, det_df in detections.groupby('oid'):
            # Convert magnitudes to nJy flux
            mag = pd.to_numeric(det_df['magpsf'], errors='coerce').values
            mag_err = pd.to_numeric(det_df['sigmapsf'], errors='coerce').values
            mag_err = np.where(np.isfinite(mag_err), mag_err, 0.05)

            valid = np.isfinite(mag) & (mag > 0) & (mag < 30)
            if not valid.any():
                continue

            flux = 10 ** ((AB_ZP_NJY - mag) / 2.5)
            flux_err = flux * np.log(10) / 2.5 * np.abs(mag_err)

            mjd = pd.to_numeric(det_df['mjd'], errors='coerce').values
            fid = det_df.get('fid', pd.Series(dtype=int))
            bands = fid.map(ztf_band_map).fillna('?').values

            lc_df = pd.DataFrame({
                'mjd': mjd, 'flux': flux, 'flux_err': flux_err,
                'magnitude': mag, 'mag_err': mag_err,
                'band': bands, 'survey': 'ZTF', 'source': 'detection',
            })
            lc_df = lc_df[valid].reset_index(drop=True)

            # Assign to all original IDs that matched this OID
            for pid in oid_to_ids.get(oid, []):
                results[pid] = lc_df

        n_with_data = len(results)
        total_detections = sum(len(df) for df in results.values())
        logger.info("ZTF batch: %d/%d candidates have ZTF data (%d total detections)",
                   n_with_data, len(positions), total_detections)
        return results

    except Exception as e:
        logger.warning("ZTF batch fetch failed: %s", e)
        import traceback
        logger.debug(traceback.format_exc())
        return {}


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
    # --- Query Fink (if available) ---
    if fink is not None:
        fink_df = fetch_fink_candidates(fink, min_sn_score=min_prob, n_fetch=n_fetch)
    else:
        logger.info("Fink API unavailable — skipping Fink candidate discovery")
        fink_df = pd.DataFrame()

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

    # --- Enrich ANTARES-only candidates with Fink classifications ---
    # ANTARES has no ML classifier; we use a heuristic proxy capped at 0.5
    # For better P(Ia), query Fink for SN classifier scores
    if fink is not None and len(screened) > 0:
        antares_only = screened[
            (screened['brokers_detected'] == 'ANTARES') &
            (screened['mean_ia_prob'] < 0.5)  # Likely using our proxy
        ].copy()

        if len(antares_only) > 0:
            logger.info("Enriching %d ANTARES-only candidates with Fink classifications...",
                        len(antares_only))
            try:
                enriched = fink.get_classifications(antares_only, radius_arcsec=2.0)
                n_enriched = enriched['fink_sn_score'].notna().sum()

                if n_enriched > 0:
                    # Update mean_ia_prob with Fink scores where available
                    for idx, row in enriched.iterrows():
                        fink_score = row.get('fink_sn_score')
                        if pd.notna(fink_score) and fink_score > 0:
                            orig_idx = screened.index[screened['diaObjectId'] == row.get('diaObjectId', row.get('object_id'))]
                            if len(orig_idx) > 0:
                                # Use Fink score instead of ANTARES proxy
                                screened.loc[orig_idx, 'sn_score'] = fink_score
                                screened.loc[orig_idx, 'mean_ia_prob'] = fink_score
                                screened.loc[orig_idx, 'fink_sn_score'] = fink_score

                    logger.info("  Updated %d candidates with Fink SN scores", n_enriched)
            except Exception as e:
                logger.warning("Fink classification enrichment failed: %s", e)

    # --- Filter by P(Ia) ---
    if 'mean_ia_prob' in screened.columns:
        before = len(screened)
        screened = screened[screened['mean_ia_prob'] >= min_prob].copy()
        logger.info("After P(Ia) >= %.2f: %d (dropped %d)",
                    min_prob, len(screened), before - len(screened))

    # --- Normalize output columns for downstream compatibility ---
    # Need: diaObjectId, ra, dec, ddf_field, sn_score
    if 'diaObjectId' not in screened.columns:
        # Build diaObjectId from available ID columns, preferring Rubin IDs
        def _get_best_id(row):
            # Priority: rubin_dia_object_id > object_id_ANTARES > object_id > unique_id > coord-based
            for col in ['rubin_dia_object_id', 'object_id_ANTARES', 'object_id_Fink',
                        'object_id_ALeRCE', 'object_id', 'unique_id']:
                if col in row.index:
                    val = row.get(col)
                    if pd.notna(val) and str(val).strip():
                        return str(val).strip()
            # Fallback: coordinate-based ID
            ra, dec = row.get('ra'), row.get('dec')
            if pd.notna(ra) and pd.notna(dec):
                return f"coord_{ra:.5f}_{dec:.5f}"
            return f"obj_{row.name}"

        screened['diaObjectId'] = screened.apply(_get_best_id, axis=1)
        logger.debug("Created diaObjectId for %d candidates", len(screened))

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


def fetch_and_fit(fink, candidates_df, mjd_now, fetch_ztf=True, fetch_atlas=True,
                  min_snr_points=5, min_bands=2, min_fit_bands=2,
                  prefilter_min_sources=0, use_salt=False, redshifts=None,
                  max_rise_time=30.0):
    """Fetch light curves from all surveys and run peak fitting for each candidate.

    Two-pass approach:
      Pass 0: (Optional) Batch query Fink source counts to pre-filter candidates.
      Pass 1: Fetch Fink/Rubin photometry for all candidates, identify bright ones.
      Batch:  Submit bright candidates (any Rubin detection < 20 mag) to ATLAS
              as a single batch, and fetch ZTF per-candidate.
      Pass 2: Combine all photometry and run fits.

    Quality cut parameters (relax for sparse early Rubin data):
      min_snr_points: Minimum points with SNR > 5 (default 5, try 3).
      min_bands: Minimum bands with detections (default 2, try 1).
      min_fit_bands: Minimum bands for successful fit (default 2, try 1).
      prefilter_min_sources: If > 0, batch pre-filter to candidates with at least
                             this many Fink sources (saves API calls). Default 0 (disabled).
      use_salt: If True and sncosmo is available, run SALT2 template fits.
      redshifts: Dict mapping diaObjectId -> redshift for SALT fitting.
      max_rise_time: Maximum rise time in days (default 30). SNe Ia rise in ~17-20d.
    """
    redshifts = redshifts or {}
    dia_ids = candidates_df['diaObjectId'].unique()
    logger.info("Fitting %d candidates...", len(dia_ids))

    # ---- Pass 0: Optional batch pre-filter by source count ----
    if prefilter_min_sources > 0 and fink is not None:
        logger.info("Pre-filtering by Fink source count (min=%d)...", prefilter_min_sources)
        candidates_df = fink.prefilter_by_source_count(
            candidates_df, min_sources=prefilter_min_sources, id_column='diaObjectId'
        )
        dia_ids = candidates_df['diaObjectId'].unique()
        if len(dia_ids) == 0:
            logger.warning("No candidates passed pre-filter — nothing to fit")
            return {}

    # Lookup RA/Dec for each candidate
    coord_lookup = {}
    for _, row in candidates_df.iterrows():
        coord_lookup[row['diaObjectId']] = (row['ra'], row['dec'])

    # MJD lower bound for ATLAS queries (avoid fetching years of old data)
    atlas_mjd_min = mjd_now - 30  # 30 days back

    # ---- Pass 1: Fetch Fink photometry and identify bright candidates ----
    FINK_MAX_CONSECUTIVE_FAILURES = 5
    fink_data = {}  # did -> DataFrame
    bright_for_atlas = []  # (did, ra, dec) for candidates brighter than cut

    if fink is None:
        logger.warning("Fink API unavailable — trying ALeRCE-LSST for Rubin photometry")
        # Fall back to ALeRCE-LSST for Rubin photometry
        alerce_lsst = None
        if HAS_ALERCE:
            try:
                alerce_lsst = AlerceClient(survey='lsst', use_db=False)
                logger.info("ALeRCE-LSST client initialized for Rubin photometry fallback")
            except Exception as e:
                logger.warning("Could not initialize ALeRCE-LSST client: %s", e)

        # Build lookup of ALeRCE-LSST OIDs for each candidate
        alerce_oid_lookup = {}
        for _, row in candidates_df.iterrows():
            did = row['diaObjectId']
            # Try ALeRCE-LSST object ID first, then the general object_id
            alerce_oid = None
            for col in ['object_id_ALeRCE-LSST', 'rubin_dia_object_id', 'object_id']:
                if col in row.index and pd.notna(row.get(col)) and str(row[col]).strip():
                    alerce_oid = str(row[col]).strip()
                    break
            if alerce_oid:
                alerce_oid_lookup[did] = alerce_oid

        if alerce_lsst is not None and alerce_oid_lookup:
            consecutive_failures = 0
            for i, did in enumerate(dia_ids):
                if consecutive_failures >= FINK_MAX_CONSECUTIVE_FAILURES:
                    break
                alerce_oid = alerce_oid_lookup.get(did)
                if not alerce_oid:
                    continue
                logger.info("[%d/%d] ALeRCE-LSST (Rubin): %s", i + 1, len(dia_ids), alerce_oid)
                try:
                    lc = alerce_lsst.get_light_curve(alerce_oid)
                    if lc is not None and len(lc) > 0:
                        fink_data[did] = lc
                        consecutive_failures = 0
                        # Check brightness for ATLAS
                        if fetch_atlas and 'magnitude' in lc.columns:
                            mags = pd.to_numeric(lc['magnitude'], errors='coerce')
                            brightest = mags.dropna().min()
                            if np.isfinite(brightest) and brightest < ATLAS_BRIGHT_MAG_CUT:
                                ra, dec = coord_lookup.get(did, (np.nan, np.nan))
                                if np.isfinite(ra) and np.isfinite(dec):
                                    bright_for_atlas.append((str(did), ra, dec))
                    else:
                        consecutive_failures += 1
                except Exception as e:
                    logger.debug("  ALeRCE-LSST error for %s: %s", alerce_oid, e)
                    consecutive_failures += 1
            logger.info("ALeRCE-LSST fallback: %d/%d candidates have Rubin photometry",
                        len(fink_data), len(dia_ids))
        else:
            logger.warning("No ALeRCE-LSST fallback available — no Rubin photometry this run")
            # Fall back to broker-reported magnitudes for ATLAS brightness cut
            if fetch_atlas:
                for _, row in candidates_df.iterrows():
                    did = row['diaObjectId']
                    mag_val = None
                    for col in ['peak_mag', 'magnitude', 'last_mag', 'meanmag']:
                        if col in row.index and pd.notna(row.get(col)):
                            mag_val = float(row[col])
                            break
                    if mag_val is not None and mag_val < ATLAS_BRIGHT_MAG_CUT:
                        ra, dec = coord_lookup.get(did, (np.nan, np.nan))
                        if np.isfinite(ra) and np.isfinite(dec):
                            bright_for_atlas.append((str(did), ra, dec))
                if bright_for_atlas:
                    logger.info("ATLAS pre-filter (from broker mags): %d candidates < %.1f mag",
                                len(bright_for_atlas), ATLAS_BRIGHT_MAG_CUT)
    else:
        consecutive_fink_failures = 0
        fink_skipped = 0
        for i, did in enumerate(dia_ids):
            # Circuit breaker: if Fink API is consistently failing, skip remaining
            if consecutive_fink_failures >= FINK_MAX_CONSECUTIVE_FAILURES:
                fink_skipped += 1
                continue

            logger.info("[%d/%d] Fink: %s", i + 1, len(dia_ids), did)
            fink_lc = fink.get_light_curve(str(did), include_forced=True)
            if fink_lc is None or len(fink_lc) == 0:
                consecutive_fink_failures += 1
                logger.warning("  No Fink light curve (consecutive failures: %d/%d)",
                               consecutive_fink_failures, FINK_MAX_CONSECUTIVE_FAILURES)
                if consecutive_fink_failures >= FINK_MAX_CONSECUTIVE_FAILURES:
                    logger.error("Fink API: %d consecutive failures — "
                                 "skipping remaining %d candidates. "
                                 "Server may be down.",
                                 FINK_MAX_CONSECUTIVE_FAILURES,
                                 len(dia_ids) - i - 1)
                continue

            # Success — reset counter
            consecutive_fink_failures = 0
            fink_data[did] = fink_lc

            # Check if any Rubin detection is brighter than the ATLAS cut
            if fetch_atlas and 'magnitude' in fink_lc.columns:
                mags = pd.to_numeric(fink_lc['magnitude'], errors='coerce')
                brightest = mags.dropna().min()
                if np.isfinite(brightest) and brightest < ATLAS_BRIGHT_MAG_CUT:
                    ra, dec = coord_lookup.get(did, (np.nan, np.nan))
                    if np.isfinite(ra) and np.isfinite(dec):
                        bright_for_atlas.append((str(did), ra, dec))

        logger.info("Fink photometry: %d/%d candidates have data%s",
                    len(fink_data), len(dia_ids),
                    f" ({fink_skipped} skipped — Fink API down)" if fink_skipped else "")

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

    # ---- Batch ZTF photometry ----
    ztf_data = {}  # did -> DataFrame (nJy format)
    if fetch_ztf and HAS_ALERCE:
        # Build position list for batch cross-match
        ztf_positions = []
        for did in dia_ids:
            ra, dec = coord_lookup.get(did, (np.nan, np.nan))
            if np.isfinite(ra) and np.isfinite(dec):
                ztf_positions.append((str(did), ra, dec))

        if ztf_positions:
            ztf_data = fetch_ztf_photometry_batch(ztf_positions, radius_arcsec=2.0)

    # ---- Pass 2: Combine photometry and fit ----
    results = {}
    for i, did in enumerate(dia_ids):
        logger.info("[%d/%d] Fitting %s", i + 1, len(dia_ids), did)
        ra, dec = coord_lookup.get(did, (np.nan, np.nan))
        fink_lc = fink_data.get(did)

        # ZTF photometry (from batch results)
        ztf_lc = ztf_data.get(str(did))

        # ATLAS photometry (from batch results)
        atlas_lc = atlas_data.get(str(did))

        # --- Combine all photometry ---
        combined = combine_photometry(fink_lc, ztf_lc, atlas_lc)
        if combined is None:
            combined = fink_lc  # fallback to Fink-only
        if combined is None or len(combined) == 0:
            logger.warning("  No photometry available — skipping")
            continue

        lc_clean = clean_light_curve(combined)

        # Quality cut: >= 5 points with SNR > 5, detections in >= 2 bands
        if 'flux' in lc_clean.columns and 'flux_err' in lc_clean.columns:
            snr = (lc_clean['flux'] / lc_clean['flux_err']).abs()
            high_snr = lc_clean[snr > 5]
        else:
            high_snr = lc_clean

        n_high_snr = len(high_snr)
        n_bands_detected = high_snr['band'].nunique() if len(high_snr) > 0 else 0
        band_counts = lc_clean.groupby('band').size()

        if n_high_snr < min_snr_points:
            logger.warning("  Too few high-SNR points (%d, need >=%d): %s",
                          n_high_snr, min_snr_points,
                          ', '.join(f"{b}={n}" for b, n in band_counts.items()))
            continue

        if n_bands_detected < min_bands:
            logger.warning("  Too few bands (%d, need >=%d): %s",
                          n_bands_detected, min_bands,
                          ', '.join(f"{b}={n}" for b, n in band_counts.items()))
            continue

        # Quality cut: require data spanning multiple nights (>= 2 day baseline)
        mjd_span = lc_clean['mjd'].max() - lc_clean['mjd'].min()
        if mjd_span < 2.0:
            logger.warning("  Single-epoch event (%.1f day span) — skipping", mjd_span)
            continue

        logger.info("  %d pts (%d SNR>5) in %d bands (%.0fd span): %s",
                    len(lc_clean), n_high_snr, n_bands_detected, mjd_span,
                    ', '.join(f"{b}={n}" for b, n in band_counts.items()))

        # --- Run multiband Villar fit (primary) and parabola (fallback) ---
        vil = fit_villar_multiband(combined)
        par = fit_parabola(combined)

        # --- Optional SALT2 template fit ---
        salt_result = None
        if use_salt and HAS_SNCOSMO:
            z = redshifts.get(did)
            salt_result = fit_salt(combined, model_name='salt2', z=z)
            if salt_result.get('status') == 'ok':
                logger.info("  SALT2: x1=%.2f, c=%.2f, chi2/dof=%.1f, z=%.3f",
                           salt_result.get('x1', np.nan),
                           salt_result.get('c', np.nan),
                           salt_result.get('chi2_dof', np.nan),
                           salt_result.get('z', np.nan))

        # Extract peak info
        vil_best = vil.get('best')
        par_best = par.get('best')
        n_bands_fit = vil.get('n_bands_fit', 0)

        # Require fit to converge in min_fit_bands — quality gate (relax for sparse data)
        if vil_best and vil_best.get('status') == 'ok' and n_bands_fit >= min_fit_bands:
            best = vil_best
            fit_method = 'villar_mb'
        elif par_best and par_best.get('status') == 'ok':
            # Accept parabola only if it fit in at least min_fit_bands
            par_bands_ok = sum(1 for b, info in par.get('per_band', {}).items()
                               if info.get('status') == 'ok')
            if par_bands_ok >= min_fit_bands:
                best = par_best
                fit_method = 'parabola'
            else:
                logger.warning("  Fits failed: Villar %s (%d bands), parabola %d bands ok",
                              vil_best.get('status', 'none') if vil_best else 'none',
                              n_bands_fit, par_bands_ok)
                continue
        else:
            logger.warning("  No acceptable multiband fit — skipping")
            continue

        peak_mag = best.get('peak_mag', np.nan)
        peak_mjd = best.get('peak_mjd', np.nan)
        peak_band = best.get('band', '')
        if not np.isfinite(peak_mag) or not np.isfinite(peak_mjd):
            logger.warning("  Fit converged but peak is NaN — skipping")
            continue

        delta_t = mjd_now - peak_mjd

        # Sanity cuts on fitted peak
        if peak_mag > 26.0:
            logger.warning("  Unphysical peak mag %.1f (>26) — skipping", peak_mag)
            continue
        if abs(delta_t) > 60:
            logger.warning("  Peak too far from now (dt=%.0fd, limit 60d) — skipping", delta_t)
            continue

        # Compute rise time (explosion to peak)
        rise_time = np.nan
        if fit_method == 'villar_mb' and vil.get('shared_t0') is not None:
            # Villar fit gives explosion epoch directly
            t0_explosion = vil.get('shared_t0')
            if np.isfinite(t0_explosion) and np.isfinite(peak_mjd):
                rise_time = peak_mjd - t0_explosion
        else:
            # Fallback: estimate from first detection
            first_mjd = lc_clean['mjd'].min()
            if np.isfinite(first_mjd) and np.isfinite(peak_mjd):
                rise_time = peak_mjd - first_mjd

        # Rise time filter: SNe Ia rise in ~17-20 days, reject slow risers
        MIN_RISE_TIME = 5.0   # days (reject if peak is before first detection)
        if np.isfinite(rise_time):
            if rise_time > max_rise_time:
                logger.warning("  Slow riser (%.1f days > %.0f) — likely not SN Ia, skipping",
                              rise_time, max_rise_time)
                continue
            if rise_time < MIN_RISE_TIME:
                logger.warning("  Unphysical rise time (%.1f days < %.0f) — bad fit, skipping",
                              rise_time, MIN_RISE_TIME)
                continue

        # Track survey coverage
        surveys_present = combined['survey'].unique().tolist() if 'survey' in combined.columns else ['Rubin']
        n_ztf = len(ztf_lc) if ztf_lc is not None else 0
        n_atlas = len(atlas_lc) if atlas_lc is not None else 0

        results[did] = {
            'diaObjectId': did,
            'parabola': par,
            'villar': vil,
            'salt': salt_result,
            'light_curve': combined,
            'light_curve_clean': lc_clean,
            'peak_mag': peak_mag,
            'peak_mjd': peak_mjd,
            'peak_band': peak_band,
            'delta_t': delta_t,
            'rise_time': rise_time,
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


def build_summary_table(candidates_df, fit_results, mjd_now, host_info=None,
                        redshifts=None):
    """Build a merged summary DataFrame with merit scores.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Candidate metadata from brokers.
    fit_results : dict
        Light curve fit results keyed by diaObjectId.
    mjd_now : float
        Current MJD for delta_t calculation.
    host_info : dict, optional
        Host galaxy info keyed by diaObjectId. Values should be dicts with
        'morphology', 'nuclear_offset_arcsec', 'offset_class', etc.
        For backwards compatibility, also accepts plain strings (morphology only).
    redshifts : dict, optional
        Redshift info keyed by diaObjectId. Values should be dicts with
        'redshift', 'distmod', 'ned_name', 'separation_arcsec' keys.
    """
    rows = []
    host_info = host_info or {}
    redshifts = redshifts or {}

    for _, cand in candidates_df.iterrows():
        did = cand['diaObjectId']
        fit = fit_results.get(did)
        if fit is None:
            continue

        peak_mag = fit['peak_mag']
        delta_t = fit['delta_t']

        # Get classifier probability and host morphology for merit calculation
        # Prefer Fink's sn_score (real ML classifier) over mean_ia_prob (may include ANTARES proxy)
        sn_score = cand.get('sn_score')
        mean_prob = cand.get('mean_ia_prob', np.nan)
        if pd.notna(sn_score) and float(sn_score) > 0:
            ia_prob = float(sn_score)
        else:
            ia_prob = mean_prob

        # Get host galaxy info (handle both dict and string formats for backwards compat)
        host_data = host_info.get(did, {})
        if isinstance(host_data, str):
            # Old format: just morphology string
            host_morph = host_data
            nuclear_offset = np.nan
            offset_class = 'unknown'
        else:
            # New format: full host info dict
            host_morph = host_data.get('morphology', 'unknown')
            nuclear_offset = host_data.get('nuclear_offset_arcsec', np.nan)
            offset_class = host_data.get('offset_class', 'unknown')

        # Get extinction and broker count for merit calculation
        extinction_ebv = cand.get('E_BV', cand.get('ebv', np.nan))
        num_brokers = cand.get('num_brokers', 1)

        # Get redshift info
        z_info = redshifts.get(did, {})
        redshift = z_info.get('redshift', np.nan) if z_info else np.nan
        distmod = z_info.get('distmod', np.nan) if z_info else np.nan
        ned_name = z_info.get('ned_name', '') if z_info else ''
        ned_sep = z_info.get('separation_arcsec', np.nan) if z_info else np.nan

        # Compute absolute magnitude if we have redshift
        absolute_mag = np.nan
        if np.isfinite(peak_mag) and np.isfinite(distmod):
            absolute_mag = peak_mag - distmod

        # Get SALT fit results
        salt = fit.get('salt')
        salt_status = salt.get('status', '') if salt else ''
        salt_x1 = salt.get('x1', np.nan) if salt and salt.get('status') == 'ok' else np.nan
        salt_c = salt.get('c', np.nan) if salt and salt.get('status') == 'ok' else np.nan
        salt_chi2_dof = salt.get('chi2_dof', np.nan) if salt and salt.get('status') == 'ok' else np.nan
        salt_z = salt.get('z', np.nan) if salt and salt.get('status') == 'ok' else np.nan
        salt_peak_mag_B = salt.get('peak_mag_B', np.nan) if salt and salt.get('status') == 'ok' else np.nan

        # Merit score with all factors (moon penalty computed later in observability filter)
        # Use breakdown version to get individual component weights
        if np.isfinite(delta_t) and np.isfinite(peak_mag):
            prob_arg = ia_prob if np.isfinite(ia_prob) else None
            ext_arg = extinction_ebv if np.isfinite(extinction_ebv) else None
            salt_arg = salt_chi2_dof if np.isfinite(salt_chi2_dof) else None
            absmag_arg = absolute_mag if np.isfinite(absolute_mag) else None
            breakdown = compute_merit_breakdown(
                delta_t, peak_mag,
                ia_prob=prob_arg,
                host_morphology=host_morph,
                extinction_ebv=ext_arg,
                num_brokers=num_brokers,
                salt_chi2_dof=salt_arg,
                absolute_mag=absmag_arg,
            )
            merit = float(breakdown['merit'])
            w_time = float(breakdown['w_time'])
            w_mag = float(breakdown['w_mag'])
            w_prob = float(breakdown['w_prob'])
            w_host = float(breakdown['w_host'])
            w_ext = float(breakdown['w_ext'])
            w_broker = float(breakdown['w_broker'])
            w_salt = float(breakdown['w_salt'])
            w_absmag = float(breakdown['w_absmag'])
        else:
            merit = np.nan
            w_time = w_mag = w_prob = w_host = w_ext = w_broker = np.nan
            w_salt = w_absmag = np.nan

        rows.append({
            'diaObjectId': did,
            'ra': cand['ra'],
            'dec': cand['dec'],
            'ddf_field': cand.get('ddf_field', ''),
            'sn_score': cand.get('sn_score', np.nan),
            'early_ia_score': cand.get('early_ia_score', np.nan),
            'brokers_detected': cand.get('brokers_detected', 'Fink'),
            'num_brokers': num_brokers,
            'mean_ia_prob': cand.get('mean_ia_prob', np.nan),
            'host_morphology': host_morph,
            'nuclear_offset_arcsec': nuclear_offset,
            'offset_class': offset_class,
            'E_BV': extinction_ebv,
            # TNS cross-match info
            'tns_name': cand.get('tns_name'),
            'tns_type': cand.get('tns_type'),
            'tns_redshift': cand.get('tns_redshift', np.nan),
            'tns_match': cand.get('tns_match', False),
            # Redshift info
            'redshift': redshift,
            'distmod': distmod,
            'ned_name': ned_name,
            'ned_sep_arcsec': ned_sep,
            'absolute_mag': absolute_mag,
            # Peak fit info
            'peak_mag': peak_mag,
            'peak_mjd': fit['peak_mjd'],
            'peak_band': fit['peak_band'],
            'delta_t': delta_t,
            'rise_time': fit.get('rise_time', np.nan),
            'fit_method': fit['fit_method'],
            'n_points': fit['n_points'],
            'n_bands': fit['n_bands'],
            'surveys': '+'.join(fit.get('surveys', ['Rubin'])),
            'n_ztf': fit.get('n_ztf', 0),
            'n_atlas': fit.get('n_atlas', 0),
            # SALT fit results
            'salt_status': salt_status,
            'salt_x1': salt_x1,
            'salt_c': salt_c,
            'salt_chi2_dof': salt_chi2_dof,
            'salt_z': salt_z,
            'salt_peak_mag_B': salt_peak_mag_B,
            # Merit breakdown
            'merit': merit,
            'w_time': w_time,
            'w_mag': w_mag,
            'w_prob': w_prob,
            'w_host': w_host,
            'w_ext': w_ext,
            'w_broker': w_broker,
            'w_salt': w_salt,
            'w_absmag': w_absmag,
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


def plot_observing_sequence_skymap(sequence_df, obs_date, ax=None):
    """Plot optimized observing sequence on a sky map.

    Shows targets color-coded by observation order (start=blue -> end=red),
    with arrows indicating slew path.

    Parameters
    ----------
    sequence_df : pd.DataFrame
        From optimize_observing_sequence(), must have ra, dec, obs_order, obs_time_ut.
    obs_date : str
        Observing date for title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig, ax : Figure and Axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if len(sequence_df) == 0:
        ax.text(0.5, 0.5, 'No targets in sequence', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return fig, ax

    df = sequence_df.sort_values('obs_order')
    n = len(df)

    # Color gradient: start (blue/purple) -> end (orange/red)
    cmap = plt.cm.plasma
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # Plot DDF field markers
    for f in DDF_FIELDS:
        ax.scatter(f['ra'], f['dec'], s=300, marker='s', facecolors='none',
                   edgecolors='lightgray', linewidths=1.5, alpha=0.5, zorder=1)
        ax.annotate(f['name'], (f['ra'], f['dec']), fontsize=7,
                    ha='center', va='bottom', alpha=0.4, xytext=(0, 8),
                    textcoords='offset points')

    # Plot slew arrows
    ras = df['ra'].values
    decs = df['dec'].values
    for i in range(n - 1):
        ax.annotate('', xy=(ras[i + 1], decs[i + 1]), xytext=(ras[i], decs[i]),
                    arrowprops=dict(arrowstyle='->', color='gray',
                                    alpha=0.4, lw=0.8),
                    zorder=2)

    # Plot targets
    sc = ax.scatter(ras, decs, c=range(n), cmap='plasma', s=80, zorder=3,
                    edgecolors='white', linewidths=0.8)

    # Add observation order labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.annotate(f"{int(row['obs_order'])}", (row['ra'], row['dec']),
                    fontsize=7, ha='center', va='center', fontweight='bold',
                    color='white', zorder=4)

    # Colorbar showing time progression
    cbar = plt.colorbar(sc, ax=ax, label='Observation Order', shrink=0.8)
    cbar.set_ticks([0, n // 2, n - 1])
    times = df['obs_time_ut'].values
    cbar.set_ticklabels([f'Start ({times[0]})', f'Mid ({times[n // 2]})',
                         f'End ({times[-1]})'])

    # Formatting
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_title(f'Optimized Observing Sequence — {obs_date}\n'
                 f'{n} targets, {df["slew_deg"].sum():.1f}° total slew')
    ax.grid(True, alpha=0.3)

    # Invert RA axis (convention: RA increases right-to-left on sky)
    ax.invert_xaxis()

    return fig, ax


def generate_pdf_report(summary_df, fit_results, plot_paths,
                        pdf_path, mjd_now, obs_date, observing_sequence=None):
    """Generate multi-page PDF report with summary and light curves."""
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_path) as pdf:
        # --- Title page ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        ut_stamp = mjd_to_utdate(mjd_now)
        ax.text(0.5, 0.60, 'SN Ia Monitoring Report',
                ha='center', va='center', fontsize=28, fontweight='bold')
        ax.text(0.5, 0.48, ut_stamp,
                ha='center', va='center', fontsize=32, fontweight='bold',
                fontfamily='monospace')
        ax.text(0.5, 0.38, f'MJD {mjd_now:.1f}  |  {obs_date}',
                ha='center', va='center', fontsize=16, fontfamily='monospace')
        ax.text(0.5, 0.34, f'{len(summary_df)} candidates with peak fits',
                ha='center', va='center', fontsize=14, color='gray')
        ax.text(0.5, 0.28, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} UT',
                ha='center', va='center', fontsize=11, color='gray')

        # Summary stats
        n_atlas = (summary_df['n_atlas'] > 0).sum() if 'n_atlas' in summary_df.columns else 0
        n_ztf = (summary_df['n_ztf'] > 0).sum() if 'n_ztf' in summary_df.columns else 0
        n_brokers = summary_df['num_brokers'].max() if 'num_brokers' in summary_df.columns else 1
        fields = summary_df['ddf_field'].nunique() if 'ddf_field' in summary_df.columns else 0
        high_merit = (summary_df['merit'] > 0.1).sum() if 'merit' in summary_df.columns else 0

        stats = (f'{fields} DDFs  |  {n_atlas} with ATLAS  |  {n_ztf} with ZTF  |  '
                 f'{high_merit} high-merit (>0.1)')
        ax.text(0.5, 0.20, stats,
                ha='center', va='center', fontsize=10, color='dimgray')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # --- Summary table page ---
        if len(summary_df) > 0:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            ax.set_title('Top 30 Candidates by Merit', fontsize=14, pad=20)

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
            display_df['diaObjectId'] = display_df['diaObjectId'].astype(str).str[-10:]

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

        # --- Merit Breakdown table page ---
        has_breakdown = all(c in summary_df.columns for c in ['w_time', 'w_mag', 'w_prob'])
        has_salt_weight = 'w_salt' in summary_df.columns
        if len(summary_df) > 0 and has_breakdown:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            ax.set_title('Merit Breakdown — Top 30 Candidates', fontsize=14, pad=20)

            # Add formula explanation
            if has_salt_weight:
                formula = 'Merit = W_time × W_mag × W_prob × W_host × W_ext × W_broker × W_salt × W_absmag'
            else:
                formula = 'Merit = W_time × W_mag × W_prob × W_host × W_ext × W_broker'
            ax.text(0.5, 0.95, formula,
                   ha='center', va='top', fontsize=9, fontfamily='monospace',
                   transform=ax.transAxes)

            table_df = summary_df.sort_values('merit', ascending=False).head(30).copy()

            breakdown_cols = ['diaObjectId', 'merit', 'w_time', 'w_mag', 'w_prob',
                             'w_host', 'w_ext', 'w_broker']
            if has_salt_weight:
                breakdown_cols.extend(['w_salt', 'w_absmag'])
            breakdown_df = table_df[[c for c in breakdown_cols if c in table_df.columns]].copy()

            # Format numbers
            for col in breakdown_df.columns:
                if col == 'diaObjectId':
                    breakdown_df[col] = breakdown_df[col].astype(str).str[-10:]
                else:
                    breakdown_df[col] = breakdown_df[col].apply(
                        lambda x: f'{x:.3f}' if pd.notna(x) and np.isfinite(x) else '--')

            # Rename columns for display
            col_names = {
                'diaObjectId': 'Object',
                'merit': 'Merit',
                'w_time': 'W_time',
                'w_mag': 'W_mag',
                'w_prob': 'W_prob',
                'w_host': 'W_host',
                'w_ext': 'W_ext',
                'w_broker': 'W_brok',
                'w_salt': 'W_salt',
                'w_absmag': 'W_abs',
            }
            breakdown_df.columns = [col_names.get(c, c) for c in breakdown_df.columns]

            tbl = ax.table(
                cellText=breakdown_df.values,
                colLabels=breakdown_df.columns,
                loc='center',
                cellLoc='center',
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7 if has_salt_weight else 8)
            tbl.auto_set_column_width(range(len(breakdown_df.columns)))
            tbl.scale(1.0, 1.4)

            # Add legend at bottom
            if has_salt_weight:
                legend_text = (
                    'W_time: Gaussian decay from peak (τ=10d)  |  W_mag: Optimal ~20.5 AB  |  W_prob: P(Ia) classifier\n'
                    'W_host: Elliptical=1.0, Spiral=0.6  |  W_ext: Galactic extinction  |  W_brok: Multi-broker bonus\n'
                    'W_salt: SALT2 chi2/dof quality [0.5-1.2]  |  W_abs: Absolute mag ~ -19.3 [0.3-1.0]'
                )
            else:
                legend_text = (
                    'W_time: Gaussian decay from peak (τ=10d)  |  '
                    'W_mag: Optimal ~20.5 AB  |  W_prob: P(Ia) classifier\n'
                    'W_host: Elliptical=1.0, Spiral=0.6  |  '
                    'W_ext: Galactic extinction  |  W_broker: Multi-broker bonus'
                )
            ax.text(0.5, 0.02, legend_text, ha='center', va='bottom',
                   fontsize=8, color='dimgray', transform=ax.transAxes)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # --- Merit Function Reference page ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Merit Function Reference', fontsize=16, fontweight='bold', pad=20)

        merit_text = """
MERIT FUNCTION

    Merit = W_time × W_mag × W_prob × W_host × W_ext × W_broker × W_salt × W_absmag

Each component ranges from 0 to ~1 (W_broker/W_salt can reach 1.2). The multiplicative
structure means a candidate needs to score well on ALL factors to rank highly.


COMPONENT DEFINITIONS

W_time  — Time from Peak: exp(−Δt² / 2τ²) with τ = 10 days
    Supernovae are most valuable for spectroscopy near peak brightness.

W_mag   — Magnitude Suitability: Gaussian at m_opt = 20.5 AB, σ = 1.5
    Penalizes targets too bright or too faint for Magellan spectroscopy.

W_prob  — Type Ia Probability: P(Ia) from ML classifier [0.1, 1.0]
    From ALeRCE or Fink. ANTARES-only use proxy capped at 0.50.

W_host  — Host Galaxy Morphology: Elliptical=1.0, Spiral=0.6, Unknown=0.7
    SNe Ia in elliptical hosts have lower Hubble diagram scatter.

W_ext   — Galactic Extinction Penalty: exp(−E(B−V) / 0.15)
    Heavily penalizes targets behind significant Milky Way dust.

W_broker — Multi-broker Agreement: 1.0 + 0.1×(N−1)
    Independent detections increase confidence. Range [1.0, 1.2].

W_salt  — SALT2 Template Fit Quality: sigmoid(chi2/dof) [0.5, 1.2]
    Good SALT2 fit (chi2/dof < 2) indicates SN Ia template match.
    Bonus for excellent fits, penalty for poor fits.

W_absmag — Absolute Magnitude: Gaussian at M_B = −19.3, σ = 0.7
    SNe Ia have M_B ~ −19.3 ± 0.5. Requires host redshift.
    Penalizes candidates with absolute mag inconsistent with SN Ia.
"""
        ax.text(0.05, 0.95, merit_text, ha='left', va='top',
                fontsize=10, fontfamily='monospace', transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # --- Observing Sequence Sky Map page ---
        if observing_sequence is not None and len(observing_sequence) > 0:
            fig, ax = plt.subplots(figsize=(11, 7))
            plot_observing_sequence_skymap(observing_sequence, obs_date, ax=ax)
            plt.tight_layout()
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
        # Sort by merit (best first) to match follow-up priority
        ordered = summary_df.sort_values('merit', ascending=False, na_position='last')
        # Build ranked list: (rank, diaObjectId) for candidates with plots
        ranked_dids = []
        for rank, (_, row) in enumerate(ordered.iterrows(), 1):
            did = row['diaObjectId']
            if did in plot_paths or str(did) in plot_paths:
                ranked_dids.append((rank, did))

        for page_start in range(0, len(ranked_dids), 4):
            page_items = ranked_dids[page_start:page_start + 4]
            n = len(page_items)
            fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n))
            if n == 1:
                axes = [axes]

            for ax, (rank, did) in zip(axes, page_items):
                path = plot_paths.get(did) or plot_paths.get(str(did))
                img = plt.imread(path)
                ax.imshow(img)
                ax.axis('off')

                # Add rank and merit annotation
                row = summary_df[summary_df['diaObjectId'] == did]
                if len(row) > 0:
                    r = row.iloc[0]
                    info = f"#{rank}"
                    if pd.notna(r['merit']):
                        info += f"  Merit={r['merit']:.3f}"
                    if pd.notna(r.get('ddf_field')):
                        info += f"  DDF={r['ddf_field']}"
                    ax.set_title(info, fontsize=9, loc='right')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    n_plot_pages = (len(ranked_dids) + 3) // 4
    n_diag_pages = 2  # scatter triptych + discovery space
    logger.info("PDF report: %s (%d pages)", pdf_path,
                2 + n_diag_pages + n_plot_pages)  # title + table + diagnostics + lightcurves


def generate_observing_schedule(plan_df, mjd_now, obs_date, output_path):
    """Generate a human-readable observing schedule ordered by priority.

    Uses exposure time estimates if available, otherwise assumes 30 min.
    Lists targets in priority order with coordinates, magnitude, merit,
    optimal observing time, and estimated exposure. Includes merit breakdown.
    """
    if len(plan_df) == 0:
        return

    # Use the priority ordering from plan_df (don't re-sort)
    df = plan_df.reset_index(drop=True)

    # Calculate total observing time from exposure estimates or 30 min default
    has_exp = 'exposure_minutes' in df.columns
    if has_exp:
        exp_vals = df['exposure_minutes'].fillna(30).values
        total_hours = exp_vals.sum() / 60
    else:
        total_hours = len(df) * 0.5

    # Check for merit breakdown columns
    has_breakdown = all(c in df.columns for c in ['w_time', 'w_mag', 'w_prob'])

    lines = []
    lines.append(f'# Magellan Observing Schedule — {obs_date} UT')
    lines.append(f'# MJD {mjd_now:.1f} | {len(df)} targets | ~{total_hours:.1f} hours total')
    lines.append(f'# Sorted by priority: time-critical > setting soon > merit')
    lines.append('#')
    lines.append(f'# {"#":>3s}  {"Object":20s}  {"RA":>11s}  {"Dec":>10s}  '
                 f'{"PkMag":>6s}  {"dt":>6s}  {"Merit":>6s}  '
                 f'{"OptUT":>5s}  {"Exp":>5s}  {"DDF":>8s}')
    lines.append(f'# {"---":>3s}  {"----":20s}  {"--":>11s}  {"---":>10s}  '
                 f'{"-----":>6s}  {"--":>6s}  {"-----":>6s}  '
                 f'{"-----":>5s}  {"---":>5s}  {"---":>8s}')

    for i, (_, row) in enumerate(df.iterrows()):
        ra_s, dec_s = radec_to_sexagesimal(row['ra'], row['dec'])

        pmag = f"{row['peak_mag']:.1f}" if np.isfinite(row.get('peak_mag', np.nan)) else '--'
        dt = f"{row['delta_t']:+.0f}d" if np.isfinite(row.get('delta_t', np.nan)) else '--'
        merit = f"{row['merit']:.3f}" if np.isfinite(row.get('merit', np.nan)) else '--'
        ddf = str(row.get('ddf_field', '') or '')
        did = str(row['diaObjectId'])[-12:]

        # Optimal observing time
        opt_ut = str(row.get('optimal_time_ut', '--') or '--')

        # Exposure time estimate
        exp_min = row.get('exposure_minutes', np.nan)
        exp_str = f"{exp_min:.0f}m" if np.isfinite(exp_min) else '30m'

        lines.append(f'  {i+1:3d}  {did:20s}  {ra_s:>11s}  {dec_s:>10s}  '
                     f'{pmag:>6s}  {dt:>6s}  {merit:>6s}  '
                     f'{opt_ut:>5s}  {exp_str:>5s}  {ddf:>8s}')

    lines.append('#')
    lines.append(f'# Total estimated observing time: ~{total_hours:.1f} hours')

    # Add moon info if available
    if 'moon_illumination' in df.columns:
        moon_illum = df['moon_illumination'].iloc[0]
        if np.isfinite(moon_illum):
            lines.append(f'# Moon illumination: {moon_illum * 100:.0f}%')

    # Add merit breakdown section
    if has_breakdown:
        has_salt_weight = 'w_salt' in df.columns
        lines.append('#')
        lines.append('# ' + '=' * 110)
        lines.append('# MERIT BREAKDOWN')
        if has_salt_weight:
            lines.append('# Merit = W_time × W_mag × W_prob × W_host × W_ext × W_broker × W_moon × W_salt × W_absmag')
        else:
            lines.append('# Merit = W_time × W_mag × W_prob × W_host × W_ext × W_broker × W_moon')
        lines.append('#   W_time  : exp(-dt²/200)      Gaussian decay from peak (tau=10d)')
        lines.append('#   W_mag   : exp(-(m-20.5)²/σ²) Optimal mag ~20.5 AB')
        lines.append('#   W_prob  : P(Ia) clipped      ML classifier probability [0.1-1.0]')
        lines.append('#   W_host  : morphology weight  Elliptical=1.0, Spiral=0.6, Unknown=0.7')
        lines.append('#   W_ext   : exp(-E(B-V)/0.15)  Galactic extinction penalty')
        lines.append('#   W_broker: 1 + 0.1*(N-1)      Multi-broker agreement bonus')
        lines.append('#   W_moon  : moon penalty       Phase/separation penalty [0.3-1.0]')
        if has_salt_weight:
            lines.append('#   W_salt  : SALT2 chi2/dof     Good template fit bonus [0.5-1.2]')
            lines.append('#   W_absmag: absolute mag       M_B ~ -19.3 consistency [0.3-1.0]')
        lines.append('# ' + '=' * 110)

        if has_salt_weight:
            lines.append(f'# {"#":>3s}  {"Object":>12s}  {"Merit":>6s}  '
                         f'{"W_time":>6s}  {"W_mag":>6s}  {"W_prob":>6s}  '
                         f'{"W_host":>6s}  {"W_ext":>6s}  {"W_brok":>6s}  '
                         f'{"W_moon":>6s}  {"W_salt":>6s}  {"W_abs":>6s}')
            lines.append(f'# {"---":>3s}  {"------":>12s}  {"-----":>6s}  '
                         f'{"------":>6s}  {"-----":>6s}  {"------":>6s}  '
                         f'{"------":>6s}  {"-----":>6s}  {"------":>6s}  '
                         f'{"------":>6s}  {"------":>6s}  {"-----":>6s}')
        else:
            lines.append(f'# {"#":>3s}  {"Object":>12s}  {"Merit":>6s}  '
                         f'{"W_time":>6s}  {"W_mag":>6s}  {"W_prob":>6s}  '
                         f'{"W_host":>6s}  {"W_ext":>6s}  {"W_brok":>6s}  {"W_moon":>6s}')
            lines.append(f'# {"---":>3s}  {"------":>12s}  {"-----":>6s}  '
                         f'{"------":>6s}  {"-----":>6s}  {"------":>6s}  '
                         f'{"------":>6s}  {"-----":>6s}  {"------":>6s}  {"------":>6s}')

        for i, (_, row) in enumerate(df.iterrows()):
            did = str(row['diaObjectId'])[-12:]
            merit = f"{row['merit']:.3f}" if np.isfinite(row.get('merit', np.nan)) else '--'
            w_time = f"{row['w_time']:.3f}" if np.isfinite(row.get('w_time', np.nan)) else '--'
            w_mag = f"{row['w_mag']:.3f}" if np.isfinite(row.get('w_mag', np.nan)) else '--'
            w_prob = f"{row['w_prob']:.3f}" if np.isfinite(row.get('w_prob', np.nan)) else '--'
            w_host = f"{row['w_host']:.3f}" if np.isfinite(row.get('w_host', np.nan)) else '--'
            w_ext = f"{row['w_ext']:.3f}" if np.isfinite(row.get('w_ext', np.nan)) else '--'
            w_broker = f"{row['w_broker']:.3f}" if np.isfinite(row.get('w_broker', np.nan)) else '--'
            w_moon = f"{row.get('moon_penalty', 1.0):.3f}" if np.isfinite(row.get('moon_penalty', np.nan)) else '1.000'

            if has_salt_weight:
                w_salt = f"{row['w_salt']:.3f}" if np.isfinite(row.get('w_salt', np.nan)) else '1.000'
                w_absmag = f"{row['w_absmag']:.3f}" if np.isfinite(row.get('w_absmag', np.nan)) else '1.000'
                lines.append(f'  {i+1:3d}  {did:>12s}  {merit:>6s}  '
                             f'{w_time:>6s}  {w_mag:>6s}  {w_prob:>6s}  '
                             f'{w_host:>6s}  {w_ext:>6s}  {w_broker:>6s}  '
                             f'{w_moon:>6s}  {w_salt:>6s}  {w_absmag:>6s}')
            else:
                lines.append(f'  {i+1:3d}  {did:>12s}  {merit:>6s}  '
                             f'{w_time:>6s}  {w_mag:>6s}  {w_prob:>6s}  '
                             f'{w_host:>6s}  {w_ext:>6s}  {w_broker:>6s}  {w_moon:>6s}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info("Observing schedule: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description='SN Ia nightly monitoring pipeline for Magellan follow-up',
        epilog='Example: python run_tonight.py 61100',
    )
    parser.add_argument('mjd', type=float, nargs='?',
                        default=Time.now().mjd,
                        help='MJD of the observing night (default: now)')
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
    parser.add_argument('--no-tns', action='store_true',
                        help='Skip TNS cross-matching')
    parser.add_argument('--fink-only', action='store_true',
                        help='Only query Fink (skip ANTARES/ALeRCE brokers)')
    parser.add_argument('--use-salt', action='store_true',
                        help='Enable SALT2 template fitting (requires sncosmo)')
    parser.add_argument('--no-redshift', action='store_true',
                        help='Skip NED redshift queries')

    # Quality cuts (relax for sparse early Rubin data)
    parser.add_argument('--min-snr-points', type=int, default=5,
                        help='Min points with SNR>5 (default: 5, try 3 for sparse data)')
    parser.add_argument('--min-bands', type=int, default=2,
                        help='Min bands with detections (default: 2, try 1 for sparse data)')
    parser.add_argument('--min-fit-bands', type=int, default=2,
                        help='Min bands for successful fit (default: 2, try 1 for sparse data)')
    parser.add_argument('--max-rise-time', type=float, default=30.0,
                        help='Max rise time in days (default: 30, SNe Ia ~17-20d)')
    parser.add_argument('--prefilter-min-sources', type=int, default=0,
                        help='Pre-filter candidates with fewer Fink sources (0=disabled, try 5)')

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

    # Add file handler so all log messages (including warnings) go to a log file
    log_path = os.path.join(night_dir, 'pipeline.log')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logger.info("Log file: %s", log_path)

    # --- Step 1: Connect to Fink ---
    fink = FinkLSSTClient()
    fink_available = fink.available
    if not fink_available:
        logger.warning("Fink LSST API not reachable — will use other brokers only")
    else:
        logger.info("Fink LSST API: connected")

    # --- Step 2: Query all brokers, merge, deduplicate, screen variables ---
    if args.fink_only and not fink_available:
        logger.error("--fink-only mode but Fink API is not reachable")
        sys.exit(1)
    if args.fink_only:
        logger.info("Mode: Fink only (--fink-only)")
    elif fink_available:
        logger.info("Mode: All brokers (Fink + ANTARES + ALeRCE-ZTF + ALeRCE-LSST)")
    else:
        logger.info("Mode: Non-Fink brokers only (ANTARES + ALeRCE-ZTF + ALeRCE-LSST)")
    candidates = fetch_all_broker_candidates(
        fink if fink_available else None,
        min_prob=args.min_prob,
        days_back=args.days_back,
        n_fetch=args.max_candidates,
        fink_only=args.fink_only,
    )
    if len(candidates) == 0:
        logger.error("No candidates found")
        sys.exit(1)

    # --- Step 2a: TNS cross-match (identify already-reported transients) ---
    do_tns = not args.no_tns if hasattr(args, 'no_tns') else True
    if do_tns and HAS_TNS:
        logger.info("TNS cross-match: checking %d candidates...", len(candidates))
        try:
            tns_client = TNSClient()
            tns_ok, tns_msg = tns_client.verify_connection()
            if tns_ok:
                candidates = tns_client.cross_match_candidates(candidates, radius_arcsec=5.0)
                n_tns_match = candidates['tns_match'].sum() if 'tns_match' in candidates.columns else 0
                n_classified = (candidates['tns_type'].notna()).sum() if 'tns_type' in candidates.columns else 0
                logger.info("TNS: %d/%d already reported, %d spectroscopically classified",
                           n_tns_match, len(candidates), n_classified)
            else:
                logger.warning("TNS cross-match: SKIPPED - %s", tns_msg)
        except Exception as e:
            logger.warning("TNS cross-match failed: %s", e)
    elif do_tns:
        logger.info("TNS cross-match: SKIPPED (tns_client not available)")

    # --- Step 3: Fetch light curves and fit peaks ---
    # ZTF: per-candidate ALeRCE queries; ATLAS: batch for bright candidates only
    do_ztf = not args.no_ztf
    do_atlas = not args.no_atlas
    if do_ztf:
        logger.info("ZTF photometry: %s", "enabled (via ALeRCE)" if HAS_ALERCE else "SKIPPED (alerce not installed)")
    if do_atlas:
        if HAS_ATLAS:
            # Verify ATLAS credentials at startup
            atlas_test = AtlasClient()
            atlas_ok, atlas_msg = atlas_test.verify_credentials()
            if atlas_ok:
                logger.info("ATLAS photometry: enabled (batch, bright < %.1f mag) - %s",
                            ATLAS_BRIGHT_MAG_CUT, atlas_msg)
            else:
                logger.warning("ATLAS photometry: DISABLED - %s", atlas_msg)
                do_atlas = False
        else:
            logger.info("ATLAS photometry: SKIPPED (atlas_client not available)")
    logger.info("Quality cuts: min_snr_points=%d, min_bands=%d, min_fit_bands=%d",
                args.min_snr_points, args.min_bands, args.min_fit_bands)
    if args.prefilter_min_sources > 0:
        logger.info("Pre-filter: enabled (min %d Fink sources)", args.prefilter_min_sources)

    # SALT fitting requires redshifts, so query NED first if SALT is requested
    do_salt = args.use_salt and HAS_SNCOSMO
    do_redshift = not args.no_redshift and HAS_NED
    if args.use_salt and not HAS_SNCOSMO:
        logger.warning("--use-salt requested but sncosmo not installed — skipping SALT fits")
    if args.use_salt:
        logger.info("SALT2 fitting: %s", "enabled" if do_salt else "disabled (sncosmo not available)")

    # --- Step 3a: Query NED redshifts (needed for SALT fitting and absolute mag) ---
    redshifts = {}  # did -> {redshift, distmod, ned_name, separation_arcsec}
    if do_redshift:
        logger.info("Querying NED for host galaxy redshifts (with caching)...")
        # Use batch function which handles caching
        from cache.alert_cache import AlertCache
        ned_cache = AlertCache(db_path='./cache/data/alert_cache.db')
        ned_df = query_ned_batch(candidates[['diaObjectId', 'ra', 'dec']].copy(),
                                  cache=ned_cache, radius_arcsec=18.0)
        # Convert to dict format for compatibility
        for _, row in ned_df.iterrows():
            did = row['diaObjectId']
            if pd.notna(row['ned_redshift']) and row['ned_redshift'] > 0:
                redshifts[did] = {
                    'redshift': row['ned_redshift'],
                    'distmod': row['ned_distmod'],
                    'ned_name': row['ned_name'],
                    'separation_arcsec': row['ned_sep_arcsec'],
                }
        logger.info("NED redshifts: %d/%d candidates have host z", len(redshifts), len(candidates))
    elif not args.no_redshift:
        logger.info("NED redshifts: SKIPPED (ned_query not available)")

    # Build redshift lookup for SALT fitting (just the z values)
    z_for_salt = {did: info['redshift'] for did, info in redshifts.items()
                  if info.get('redshift') is not None and info['redshift'] > 0}

    fit_results = fetch_and_fit(fink if fink_available else None,
                                candidates, mjd_now,
                                fetch_ztf=do_ztf and HAS_ALERCE,
                                fetch_atlas=do_atlas and HAS_ATLAS,
                                min_snr_points=args.min_snr_points,
                                min_bands=args.min_bands,
                                min_fit_bands=args.min_fit_bands,
                                prefilter_min_sources=args.prefilter_min_sources,
                                use_salt=do_salt,
                                redshifts=z_for_salt,
                                max_rise_time=args.max_rise_time)
    if not fit_results:
        logger.error("No successful fits")
        sys.exit(1)
    logger.info("Successful fits: %d / %d", len(fit_results), len(candidates))

    # --- Step 4: Host galaxy morphology classification + nuclear offset ---
    host_info = {}  # did -> {morphology, nuclear_offset_arcsec, offset_class, ...}
    if HAS_MORPHOLOGY:
        logger.info("Classifying host galaxy morphologies for %d candidates...",
                    len(fit_results))
        morph_filter = MorphologyFilter(cache_dir='./cache/data')
        morph_counts = {'elliptical': 0, 'spiral': 0, 'uncertain': 0, 'unknown': 0}
        offset_counts = {'nuclear': 0, 'offset': 0, 'distant': 0, 'unknown': 0}
        n_processed = 0
        n_no_match = 0

        for did in fit_results.keys():
            cand_row = candidates[candidates['diaObjectId'] == did]
            if len(cand_row) == 0:
                n_no_match += 1
                host_info[did] = {'morphology': 'unknown', 'offset_class': 'unknown'}
                continue

            ra, dec = float(cand_row.iloc[0]['ra']), float(cand_row.iloc[0]['dec'])
            try:
                info = morph_filter.classify_host_galaxy(ra, dec)
                morph = info.get('morphology', 'unknown')
                offset_class = info.get('offset_class', 'unknown')

                host_info[did] = info
                morph_counts[morph] = morph_counts.get(morph, 0) + 1
                offset_counts[offset_class] = offset_counts.get(offset_class, 0) + 1
                n_processed += 1

                # Log progress every 10 candidates
                if n_processed % 10 == 0:
                    logger.info("  Morphology progress: %d/%d processed",
                                n_processed, len(fit_results))

            except Exception as e:
                logger.warning("Host morphology query failed for %s at (%.3f, %.3f): %s",
                               did, ra, dec, e)
                host_info[did] = {'morphology': 'unknown', 'offset_class': 'unknown'}
                morph_counts['unknown'] += 1

        # Summary breakdown
        logger.info("Host morphology complete: %d elliptical, %d spiral, "
                    "%d uncertain, %d unknown (+ %d no coord match)",
                    morph_counts['elliptical'], morph_counts['spiral'],
                    morph_counts['uncertain'], morph_counts['unknown'], n_no_match)
        logger.info("Nuclear offset: %d nuclear (<1\"), %d offset (1-30\"), "
                    "%d distant (>30\"), %d unknown",
                    offset_counts['nuclear'], offset_counts['offset'],
                    offset_counts['distant'], offset_counts['unknown'])
        if offset_counts['nuclear'] > 0:
            logger.warning("  *** %d candidates are NUCLEAR (potential AGN/TDE) ***",
                          offset_counts['nuclear'])
    else:
        logger.info("Host morphology: SKIPPED (morphology_filter not available)")

    # --- Step 5: Build summary table with merit scores ---
    summary = build_summary_table(candidates, fit_results, mjd_now, host_info,
                                  redshifts=redshifts)
    logger.info("Summary table: %d rows", len(summary))

    if len(summary) == 0:
        logger.error("Empty summary table")
        sys.exit(1)

    # --- Step 6: Observability filter ---
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

    # Prioritize targets by time-criticality, setting time, and merit
    plan = prioritize_targets(plan)
    plan = plan.reset_index(drop=True)
    logger.info("Observing plan: %d targets (priority-ordered)", len(plan))

    # Generate optimized single-night observing sequence (minimizes slew)
    logger.info("Computing optimized observing sequence...")
    observing_sequence = optimize_observing_sequence(
        plan, obs_date,
        max_targets=min(20, len(plan)),  # Realistic for one night
        slew_weight=0.4,
        merit_weight=0.6,
        exposure_minutes=30,
    )
    logger.info("Optimized sequence: %d targets, %.1f deg total slew",
                len(observing_sequence),
                observing_sequence['slew_deg'].sum() if len(observing_sequence) > 0 else 0)

    # --- Step 7: Generate light curve plots ---
    plot_paths = generate_light_curve_plots(fit_results, lc_dir, summary)

    # --- Step 8: Save outputs ---
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
    pdf_path = os.path.join(night_dir, f'report_{ut_stamp}.pdf')
    generate_pdf_report(summary, fit_results, plot_paths,
                        pdf_path, mjd_now, obs_date,
                        observing_sequence=observing_sequence)

    # --- Done ---
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Night directory: %s", night_dir)
    logger.info("  candidates.csv         %d candidates", len(summary))
    logger.info("  magellan_plan.cat      %d targets (priority-ordered)", len(plan))
    logger.info("  observing_schedule.txt readable schedule")
    logger.info("  report_%s.pdf       summary + light curves", ut_stamp)
    logger.info("  lightcurves/           %d plots", len(plot_paths))

    # Survey coverage summary
    if 'surveys' in summary.columns:
        n_with_ztf = (summary['n_ztf'] > 0).sum() if 'n_ztf' in summary.columns else 0
        n_with_atlas = (summary['n_atlas'] > 0).sum() if 'n_atlas' in summary.columns else 0
        logger.info("  Survey coverage: %d with ZTF, %d with ATLAS", n_with_ztf, n_with_atlas)

    # TNS cross-match summary
    if 'tns_match' in summary.columns:
        n_in_tns = summary['tns_match'].sum()
        n_tns_classified = summary['tns_type'].notna().sum()
        n_tns_snia = (summary['tns_type'].str.contains('Ia', case=False, na=False)).sum()
        if n_in_tns > 0:
            logger.info("  TNS status: %d/%d already reported, %d spectroscopically classified (%d SN Ia)",
                       n_in_tns, len(summary), n_tns_classified, n_tns_snia)

    # Nuclear offset summary (AGN/TDE screening)
    if 'offset_class' in summary.columns:
        n_nuclear = (summary['offset_class'] == 'nuclear').sum()
        n_offset = (summary['offset_class'] == 'offset').sum()
        n_distant = (summary['offset_class'] == 'distant').sum()
        if n_nuclear > 0:
            logger.warning("  Nuclear offset: %d NUCLEAR (likely AGN/TDE), %d offset (SN-like), %d distant",
                          n_nuclear, n_offset, n_distant)
            # List nuclear candidates for attention
            nuclear_cands = summary[summary['offset_class'] == 'nuclear']['diaObjectId'].tolist()
            logger.warning("    Nuclear candidates: %s", ', '.join(str(c) for c in nuclear_cands[:5]))

    # Redshift and SALT summary
    if 'redshift' in summary.columns:
        n_with_z = summary['redshift'].notna().sum()
        if n_with_z > 0:
            z_median = summary['redshift'].dropna().median()
            logger.info("  Redshifts: %d candidates with z (median z=%.3f)", n_with_z, z_median)
    if 'salt_status' in summary.columns:
        n_salt_ok = (summary['salt_status'] == 'ok').sum()
        if n_salt_ok > 0:
            good_chi2 = summary[(summary['salt_status'] == 'ok') & (summary['salt_chi2_dof'] < 2)]
            logger.info("  SALT2 fits: %d successful (%d with chi2/dof < 2)",
                       n_salt_ok, len(good_chi2))

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

    # Print priority-ordered schedule summary
    if len(plan) > 0:
        has_exp = 'exposure_minutes' in plan.columns
        total_exp = plan['exposure_minutes'].sum() / 60 if has_exp else len(plan) * 0.5
        logger.info("\nPriority-ordered schedule (%d targets, ~%.1f hours):",
                    len(plan), total_exp)
        for i, (_, r) in enumerate(plan.iterrows()):
            ra_s, _ = radec_to_sexagesimal(r['ra'], r['dec'])
            exp_str = f"{r['exposure_minutes']:.0f}m" if has_exp and np.isfinite(r.get('exposure_minutes', np.nan)) else "30m"
            opt_ut = r.get('optimal_time_ut', '--') or '--'
            logger.info("  %2d. %s  RA=%s  mag=%.1f  merit=%.3f  exp=%s  opt=%s",
                        i + 1, str(r['diaObjectId'])[-12:], ra_s,
                        r['peak_mag'] if np.isfinite(r['peak_mag']) else 0,
                        r['merit'] if np.isfinite(r['merit']) else 0,
                        exp_str, opt_ut)

    # Print optimized observing sequence (slew-minimized)
    if len(observing_sequence) > 0:
        total_slew = observing_sequence['slew_deg'].sum()
        logger.info("\nOptimized single-night sequence (%d targets, %.1f deg total slew):",
                    len(observing_sequence), total_slew)
        for _, r in observing_sequence.iterrows():
            ra_s, _ = radec_to_sexagesimal(r['ra'], r['dec'])
            logger.info("  %2d. %s  UT=%s  RA=%s  slew=%4.1f°  merit=%.3f",
                        int(r['obs_order']), str(r['diaObjectId'])[-12:],
                        r['obs_time_ut'], ra_s,
                        r['slew_deg'], r['merit'] if np.isfinite(r['merit']) else 0)

        # Save optimized sequence to file
        seq_path = os.path.join(night_dir, 'optimized_sequence.csv')
        seq_cols = ['obs_order', 'obs_time_ut', 'diaObjectId', 'ra', 'dec',
                    'peak_mag', 'merit', 'slew_deg', 'cumulative_time_hr', 'ddf_field']
        observing_sequence[[c for c in seq_cols if c in observing_sequence.columns]].to_csv(
            seq_path, index=False)
        logger.info("Optimized sequence saved: %s", seq_path)


if __name__ == '__main__':
    main()
