#!/usr/bin/env python3
"""Nightly SN Ia monitoring pipeline for Magellan follow-up.

Usage:
    python run_tonight.py <MJD>
    python run_tonight.py 61100
    python run_tonight.py 61100 --min-prob 0.3 --days-back 30

Creates a night directory (e.g., nights/ut20260301/) containing:
    - candidates.csv          Ranked candidate table (the pipeline's product)
    - llamas/                 Executable LLAMAS plan from the orchestrator —
                              timeline, TCS catalog, summary, time accounting
                              (the single scheduling authority)
    - report_{ut_stamp}.pdf   Multi-page PDF with light curves and summary
    - lightcurves/            Per-candidate magnitude plots (PNG)

The pipeline RANKS; the LLAMAS orchestrator SCHEDULES. The pipeline's own
schedule/catalog/sequence outputs were retired 2026-07 (duplicate scheduling
implementations with divergent rules).
"""

import argparse
import logging
import json
import os
import sys
import time
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
    load_salt_model, salt_z_policy, choose_best_fit,
)
# NOTE: write_magellan_catalog / prioritize_targets /
# optimize_observing_sequence are no longer imported — the pipeline's own
# scheduling tail was retired 2026-07 in favor of the LLAMAS orchestrator
# (single scheduling authority; see Step 9 in main()).
from core.magellan_planning import (
    compute_merit, compute_merit_breakdown, filter_observable_targets,
    radec_to_sexagesimal, EXOTIC_PROFILE,
)
from core.ddf_fields import DDF_FIELDS, is_in_ddf, max_possible_brokers
from core.fink_breaker import (
    FinkBreaker, ACTION_FETCH, ACTION_COOLDOWN, ACTION_PROCEED,
    FINK_MAX_CONSECUTIVE_FAILURES, FINK_MAX_COOLDOWNS, FINK_COOLDOWN_SECONDS,
)

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


def fetch_fink_candidates(fink, min_sn_score=0.3, n_fetch=500, sky_mode='ddf'):
    """Fetch SN candidates from Fink and format for the multi-broker merger.

    In ``wide`` sky mode the bright-transient tag is added and the returned
    frame carries payload-level selection columns (magnitude, photo-z, TNS
    cross-match, variable-star/AGN cross-matches) so that
    ``select_wide_candidates`` can cut cheaply *before* any per-object work.
    """
    logger.info("Querying Fink LSST API...")

    tags = ['sn_near_galaxy_candidate', 'extragalactic_new_candidate']
    if sky_mode == 'wide':
        # Bright extragalactic stream: exactly the wide-mode target population
        tags.append('extragalactic_lt20mag_candidate')

    frames = []
    for tag in tags:
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

    # --- Payload-level enrichment (selection inputs; one row per alert) ---
    def _num(col):
        return pd.to_numeric(raw.get(col, pd.Series(dtype=float, index=raw.index)),
                             errors='coerce')

    psf = _num('r:psfFlux')  # nJy
    with np.errstate(invalid='ignore', divide='ignore'):
        raw['mag_ab'] = 31.4 - 2.5 * np.log10(psf.where(psf > 0))
    raw['alert_mjd'] = _num('r:midpointMjdTai')
    raw['z_phot'] = _num('f:xm_legacydr8_zphot')
    raw['z_tns'] = _num('f:xm_tns_redshift')
    for src, dst in [('f:xm_tns_fullname', 'tns_name_xm'),
                     ('f:xm_tns_type', 'tns_type_xm'),
                     ('f:xm_gcvs_type', 'gcvs_type_xm'),
                     ('f:xm_vsx_Type', 'vsx_type_xm'),
                     ('f:xm_simbad_otype', 'simbad_otype_xm'),
                     ('f:xm_gaiadr3_VarFlag', 'gaia_varflag_xm')]:
        raw[dst] = raw.get(src, pd.Series('', index=raw.index, dtype=object))
    raw['gaia_plx'] = _num('f:xm_gaiadr3_Plx')
    raw['gaia_plx_err'] = _num('f:xm_gaiadr3_e_Plx')

    raw['diaObjectId'] = raw['r:diaObjectId'].astype(str)

    # Per-object aggregates across ALL alerts of the object (pre score filter):
    # the brightest detection tells us if it is followable; the latest tells
    # us if it is fresh.
    grp = raw.groupby('diaObjectId')
    brightest = grp['mag_ab'].min()
    latest = grp['alert_mjd'].max()
    best_score = grp['sn_score'].max()
    best_early_ia = grp['early_ia_score'].max()

    # Filter by SN score (object passes if ANY of its alerts scored well)
    raw['_obj_best_score'] = raw['diaObjectId'].map(best_score)
    good = raw[raw['_obj_best_score'] >= min_sn_score].copy()

    # Deduplicate by diaObjectId, keeping the MOST RECENT alert (freshest
    # magnitude/state); per-object best scores are re-attached below.
    good = good.sort_values('alert_mjd', ascending=False)
    good = good.drop_duplicates(subset='diaObjectId', keep='first')
    good['sn_score'] = good['diaObjectId'].map(best_score)
    good['early_ia_score'] = good['diaObjectId'].map(best_early_ia)
    good['brightest_mag'] = good['diaObjectId'].map(brightest)
    good['last_mjd'] = good['diaObjectId'].map(latest)

    # Cross-match columns: an xm hit (TNS report, GCVS/VSX entry, ...) may be
    # attached to earlier alerts of an object and blank on the freshest one —
    # the kept (most recent) row must not lose it. Take the most-recent
    # NON-BLANK value per object instead. Most-recent ordering is the right
    # tiebreak: TNS xm appears on alerts only after the TNS report exists.
    raw_recent = raw.sort_values('alert_mjd', ascending=False)
    grp_recent = raw_recent.groupby('diaObjectId')

    def _first_nonblank(s):
        for v in s:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if str(v).strip() != '':
                return v
        return ''

    for col in ['tns_name_xm', 'tns_type_xm', 'gcvs_type_xm', 'vsx_type_xm',
                'simbad_otype_xm', 'gaia_varflag_xm']:
        good[col] = good['diaObjectId'].map(grp_recent[col].apply(_first_nonblank))
    # Numeric cross-match values: most-recent finite value per object. (Gaia
    # plx/err come from the same catalog match, so they stay consistent.)
    for col in ['z_tns', 'z_phot', 'gaia_plx', 'gaia_plx_err']:
        first_valid = grp_recent[col].apply(
            lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        good[col] = good['diaObjectId'].map(first_valid)

    good = good.drop(columns=['_obj_best_score'])

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


# Simbad otypes that indicate a persistent nuclear source, not a SN
_AGN_OTYPE_KEYWORDS = ('qso', 'agn', 'sy1', 'sy2', 'seyfert', 'blazar',
                       'bl lac', 'bllac', 'liner')

# Wide-mode selection defaults (Stubbs 2026B proposal target space)
WIDE_DEC_LIMIT = 22.0        # deg; airmass 1.6 at transit from LCO (lat -29.01)
WIDE_MAX_MAG = 21.5          # proposal: 18.0 <= r <= 21.5
WIDE_MAX_Z = 0.4             # proposal: 0.1 < z < 0.4
WIDE_HOSTLESS_MAX_MAG = 20.5 # keep no-redshift objects only if clearly bright
WIDE_FIT_CAP = 150           # max objects sent to per-object photometry+fit
WIDE_MIN_GAL_B = 10.0        # deg; |b| below this is nova/CV territory, not SNe


def select_wide_candidates(df, mjd_now, days_back=30,
                           dec_limit=WIDE_DEC_LIMIT, max_mag=WIDE_MAX_MAG,
                           max_z=WIDE_MAX_Z,
                           hostless_max_mag=WIDE_HOSTLESS_MAX_MAG,
                           fit_cap=WIDE_FIT_CAP, min_gal_b=WIDE_MIN_GAL_B):
    """Payload-level selection for wide (non-DDF-restricted) sky mode.

    Applies the proposal's target-space cuts using columns already present in
    the Fink alert payload — magnitude (psfFlux), Legacy DR8 photo-z / TNS
    spec-z, declination, recency, and variable-star/AGN cross-matches — so
    that expensive per-object photometry fetching and light-curve fitting only
    runs on objects we could actually follow up. The DDF variable catalogs do
    not cover the wide sky, so the payload cross-matches (GCVS, VSX, Simbad,
    Gaia) stand in as the variable screen here.

    Returns the selected frame with ``z_best``/``z_source`` columns, sorted by
    SN score and capped at ``fit_cap`` rows.
    """
    if len(df) == 0:
        return df
    n0 = len(df)

    def _blank(col):
        s = df.get(col, pd.Series('', index=df.index, dtype=object))
        return s.fillna('').astype(str).str.strip()

    # 1. Magellan-visible declination + Galactic-plane rejection: a "bright
    #    SN candidate" at |b| < ~10 deg is overwhelmingly a nova/CV/YSO, and
    #    extinction ruins it for cosmology regardless.
    ra_num = pd.to_numeric(df['ra'], errors='coerce')
    dec_num = pd.to_numeric(df['dec'], errors='coerce')
    coords_ok = ra_num.notna() & dec_num.notna()
    gal_b = pd.Series(np.nan, index=df.index)
    if coords_ok.any():
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        sc = SkyCoord(ra=ra_num[coords_ok].values * u.deg,
                      dec=dec_num[coords_ok].values * u.deg)
        gal_b.loc[coords_ok] = sc.galactic.b.deg
    df = df[(dec_num <= dec_limit) & (gal_b.abs() >= min_gal_b)]
    n_dec = len(df)

    # 2. Freshness: last alert within days_back (finally makes --days-back
    #    real for the Fink stream; unknown MJD is kept, not fabricated-fresh)
    last = pd.to_numeric(df.get('last_mjd', pd.Series(dtype=float, index=df.index)),
                         errors='coerce')
    df = df[(last >= mjd_now - days_back) | last.isna()]
    n_fresh = len(df)

    # 3. Payload variable/AGN screen (wide-sky stand-in for the DDF catalogs)
    gcvs = _blank('gcvs_type_xm')
    vsx = _blank('vsx_type_xm')
    otype = _blank('simbad_otype_xm').str.lower()
    gaia_var = _blank('gaia_varflag_xm').str.upper()
    plx = pd.to_numeric(df.get('gaia_plx', pd.Series(dtype=float, index=df.index)),
                        errors='coerce')
    plx_err = pd.to_numeric(df.get('gaia_plx_err', pd.Series(dtype=float, index=df.index)),
                            errors='coerce')
    is_var_star = (gcvs.reindex(df.index, fill_value='') != '') | \
                  (vsx.reindex(df.index, fill_value='') != '') | \
                  (gaia_var.reindex(df.index, fill_value='') == 'VARIABLE')
    is_agn = otype.reindex(df.index, fill_value='').apply(
        lambda t: any(k in t for k in _AGN_OTYPE_KEYWORDS)).astype(bool)
    with np.errstate(invalid='ignore', divide='ignore'):
        is_star = (plx / plx_err) > 5.0  # significant Gaia parallax
    is_star = is_star.fillna(False)
    df = df[~(is_var_star | is_agn | is_star)]
    n_screen = len(df)

    # 4. Brightness: the object's brightest detection must be followable
    bmag = pd.to_numeric(df.get('brightest_mag', pd.Series(dtype=float, index=df.index)),
                         errors='coerce')
    df = df[bmag <= max_mag]
    n_bright = len(df)

    # 5. Redshift: TNS spec-z preferred, else Legacy photo-z. Objects with no
    #    redshift at all are kept only if clearly bright (a nearby SN in a
    #    faint/uncatalogued host must not be discarded), and flagged.
    z_tns = pd.to_numeric(df.get('z_tns', pd.Series(dtype=float, index=df.index)),
                          errors='coerce')
    z_phot = pd.to_numeric(df.get('z_phot', pd.Series(dtype=float, index=df.index)),
                           errors='coerce')
    z_best = z_tns.where(z_tns.notna(), z_phot)
    df = df.copy()
    df['z_best'] = z_best
    df['z_source'] = np.where(z_tns.notna(), 'tns_specz',
                              np.where(z_phot.notna(), 'legacy_photz', 'none'))
    bmag = pd.to_numeric(df['brightest_mag'], errors='coerce')
    keep = (df['z_best'] <= max_z) | (df['z_best'].isna() & (bmag <= hostless_max_mag))
    df = df[keep]
    n_z = len(df)

    # 6. Cap for per-object runtime, best ML score first
    df = df.sort_values('sn_score', ascending=False)
    dropped = max(0, len(df) - fit_cap)
    df = df.head(fit_cap)

    logger.info("Wide selection: %d -> dec<=%+.0f,|b|>=%.0f: %d -> fresh(%dd): %d -> "
                "var/AGN screen: %d -> r<=%.1f: %d -> z<=%.1f|hostless<=%.1f: %d"
                "%s",
                n0, dec_limit, min_gal_b, n_dec, days_back, n_fresh, n_screen,
                max_mag, n_bright, max_z, hostless_max_mag, n_z,
                f" -> capped at {fit_cap} (dropped {dropped})" if dropped else "")
    return df


def fetch_ztf_wide_candidates(mjd_now, min_prob=0.3, days_back=30,
                              dec_limit=WIDE_DEC_LIMIT, max_mag=WIDE_MAX_MAG,
                              hostless_max_mag=WIDE_HOSTLESS_MAX_MAG,
                              max_baseline_days=150.0,
                              min_gal_b=WIDE_MIN_GAL_B):
    """Fetch live ZTF SN candidates from ALeRCE for wide sky mode.

    Uses the time-filtered multi-classifier union
    ``query_fresh_sn_candidates_multi`` (fresh, short baseline,
    Magellan-visible at the SQL level; BHRF forced-phot + legacy
    lc_classifier pools, per-object provenance in ``alerce_classifier``)
    rather than the legacy arbitrary-slice query, then applies the
    Galactic-plane screen and the brightness rule. ALeRCE carries no
    redshift, so — mirroring the Fink hostless rule — z-less ZTF objects
    are kept only if brighter than ``hostless_max_mag``. The classifier
    class tag (SNIa/SNIbc/SNII/SLSN) is carried through in ``alerce_class``
    for program-specific ranking; the selection does NOT filter to Ia.

    Returns (DataFrame in the wide-candidate schema, status dict).
    """
    status = {'queried': True, 'responded': False, 'n_returned': 0,
              'error': None}
    try:
        from broker_clients.alerce_db_client import AlerceDBClient
        db = AlerceDBClient()
        db.connect()
        rows = db.query_fresh_sn_candidates_multi(
            mjd_now, min_prob=min_prob, days_back=days_back,
            max_baseline_days=max_baseline_days, dec_limit=dec_limit)
    except Exception as e:
        status['error'] = str(e)
        logger.warning("ALeRCE-ZTF wide query failed: %s", e)
        return pd.DataFrame(), status

    status['responded'] = True
    if len(rows) == 0:
        return pd.DataFrame(), status

    # Aggregate (object, band) rows -> one row per object
    if 'alerce_classifier' not in rows.columns:
        rows = rows.copy()
        rows['alerce_classifier'] = 'lc_classifier'
    g = rows.groupby('oid').agg(
        ra=('meanra', 'first'), dec=('meandec', 'first'),
        firstmjd=('firstmjd', 'first'), deltajd=('deltajd', 'first'),
        alerce_class=('class_name', 'first'),
        probability=('probability', 'max'),
        brightest_mag=('magmin', 'min'),
        alerce_classifier=('alerce_classifier', 'first'),
    ).reset_index()
    g['last_mjd'] = g['firstmjd'] + g['deltajd']

    # Galactic-plane screen (same rationale as select_wide_candidates)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sc = SkyCoord(ra=g['ra'].values * u.deg, dec=g['dec'].values * u.deg)
    g = g[np.abs(sc.galactic.b.deg) >= min_gal_b]

    # Brightness: no redshift available from ALeRCE, so apply the same
    # hostless rule as the Fink path — bright enough to follow blind.
    bmag = pd.to_numeric(g['brightest_mag'], errors='coerce')
    g = g[bmag <= min(max_mag, hostless_max_mag)]

    # Normalize to the wide-candidate schema (aggregator input)
    g = g.copy()
    g['diaObjectId'] = g['oid'].astype(str)
    g['object_id'] = g['diaObjectId']
    # sn_score: the raw lc_classifier class probability for ALL classes — a
    # selection/confidence signal, whatever the class.
    g['sn_score'] = g['probability']
    # sn_ia_prob: ONLY meaningful as a Ia probability when the classified
    # class IS SNIa. For other classes the probability says "probably a
    # SNII", which must never be pooled into mean_ia_prob as if it were
    # P(Ia) — leave it NaN so the aggregator skips it.
    g['sn_ia_prob'] = np.where(g['alerce_class'] == 'SNIa',
                               g['probability'], np.nan)
    g['early_ia_score'] = np.nan
    g['broker'] = 'ALeRCE-ZTF'
    g['z_phot'] = np.nan
    g['z_tns'] = np.nan
    g['ddf_field'] = g.apply(
        lambda r: is_in_ddf(r['ra'], r['dec']), axis=1)

    status['n_returned'] = len(g)
    logger.info("ALeRCE-ZTF wide: %d live candidates after |b|>=%.0f and "
                "mag<=%.1f (classes: %s; classifiers: %s)",
                len(g), min_gal_b, min(max_mag, hostless_max_mag),
                dict(g['alerce_class'].value_counts()),
                dict(g['alerce_classifier'].value_counts()))
    return g, status


def fetch_finkztf_wide_candidates(mjd_now, min_prob=0.3, days_back=30,
                                  dec_limit=WIDE_DEC_LIMIT, max_mag=WIDE_MAX_MAG,
                                  hostless_max_mag=WIDE_HOSTLESS_MAX_MAG,
                                  min_gal_b=WIDE_MIN_GAL_B):
    """Fetch live ZTF SN candidates from Fink for wide sky mode.

    The downtime-resilient twin of :func:`fetch_ztf_wide_candidates`: same
    wide-candidate schema and the same screens (Galactic-plane, hostless
    brightness), but sourced from Fink's ZTF portal (France, live) instead of
    the Chile-hosted ALeRCE DB (stale during the 2026-08 storm). It is a pure
    top-of-funnel swap — the returned frame slots into the same
    ``merge_alerts`` call, so everything downstream is identical.

    Fink's SNN Ia-vs-nonIa score is a genuine P(Ia), so it populates
    ``sn_ia_prob`` directly (unlike ALeRCE, whose class probability is only
    P(Ia) when the class IS SNIa). Non-Ia SNe are kept via ``sn_score`` exactly
    as in the ALeRCE path. Returns (DataFrame in the wide-candidate schema,
    status dict).
    """
    status = {'queried': True, 'responded': False, 'n_returned': 0,
              'error': None}
    try:
        from broker_clients.fink_ztf_client import FinkZTFClient
        client = FinkZTFClient()
        # min_ia_score=0: keep all SN candidates (non-Ia survive on sn_score,
        # mirroring ALeRCE); the effective-prob filter downstream applies min_prob.
        cand = client.fetch_fresh_sn_candidates(
            mjd_now, days_back=days_back,
            max_mag=min(max_mag, hostless_max_mag), dec_max=dec_limit,
            min_ia_score=0.0)
    except Exception as e:
        status['error'] = str(e)
        logger.warning("Fink-ZTF wide query failed: %s", e)
        return pd.DataFrame(), status

    status['responded'] = True
    if cand is None or len(cand) == 0:
        return pd.DataFrame(), status

    # Galactic-plane screen (same rationale/threshold as the ALeRCE-ZTF path)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    g = cand.copy()
    sc = SkyCoord(ra=g['ra'].values * u.deg, dec=g['dec'].values * u.deg)
    g = g[np.abs(sc.galactic.b.deg) >= min_gal_b].copy()

    # Normalize to the wide-candidate schema (aggregator input) — identical
    # column set to fetch_ztf_wide_candidates so the merge/downstream is uniform.
    g['diaObjectId'] = g['objectId'].astype(str)
    g['object_id'] = g['diaObjectId']
    g['sn_ia_prob'] = pd.to_numeric(g['ia_score'], errors='coerce')
    g['sn_score'] = pd.to_numeric(g['sn_score'], errors='coerce')
    g['early_ia_score'] = np.where(g['fink_class'] == 'Early SN Ia candidate',
                                   g['sn_ia_prob'], np.nan)
    g['broker'] = 'Fink-ZTF'
    g['z_phot'] = np.nan
    g['z_tns'] = np.nan
    g['brightest_mag'] = pd.to_numeric(g['magnitude'], errors='coerce')
    g['last_mjd'] = pd.to_numeric(g['mjd'], errors='coerce')
    g['ddf_field'] = g.apply(lambda r: is_in_ddf(r['ra'], r['dec']), axis=1)

    status['n_returned'] = len(g)
    logger.info("Fink-ZTF wide: %d live candidates after |b|>=%.0f and mag<=%.1f "
                "(classes: %s; %d TNS-classified)",
                len(g), min_gal_b, min(max_mag, hostless_max_mag),
                dict(g['fink_class'].value_counts()),
                int(g.get('tns_classified', pd.Series(dtype=bool)).sum()))
    return g, status


def format_broker_status_lines(broker_status, prefix=''):
    """Return a list of human-readable lines describing per-broker liveness.

    A broker that was queried but did not respond (raised / fell back) is a
    silent-failure warning: an empty sky from a down broker must never look
    like "no SNe tonight".
    """
    lines = []
    lines.append(f'{prefix}Broker Status')
    lines.append(f'{prefix}{"Broker":16s}  {"Queried":>7s}  {"Resp":>5s}  '
                 f'{"NReturned":>9s}  Error')
    lines.append(f'{prefix}{"-" * 16}  {"-" * 7}  {"-" * 5}  {"-" * 9}  {"-" * 20}')
    if not broker_status:
        lines.append(f'{prefix}(no broker status recorded)')
        return lines
    for broker in sorted(broker_status):
        st = broker_status[broker] or {}
        q = 'yes' if st.get('queried') else 'no'
        r = 'yes' if st.get('responded') else 'NO'
        n = st.get('n_returned', 0)
        err = st.get('error') or ''
        lines.append(f'{prefix}{broker:16s}  {q:>7s}  {r:>5s}  {n:>9d}  {err}')
    return lines


def log_broker_status(broker_status):
    """Log the broker-liveness block, warning on any queried-but-unresponsive."""
    for line in format_broker_status_lines(broker_status):
        logger.info(line)
    down = [b for b, s in (broker_status or {}).items()
            if s and s.get('queried') and not s.get('responded')]
    if down:
        logger.warning("Brokers queried but did not respond: %s "
                       "(empty results may be a silent failure, not an empty sky)",
                       ', '.join(sorted(down)))


def write_broker_status(broker_status, night_dir):
    """Write a broker_status.json sidecar into the night output directory."""
    os.makedirs(night_dir, exist_ok=True)
    path = os.path.join(night_dir, 'broker_status.json')
    with open(path, 'w') as f:
        json.dump(broker_status or {}, f, indent=2, sort_keys=True)
    logger.info("Broker status: %s", path)
    return path


def _coalesce_effective_prob(screened, min_prob):
    """Coalesced effective probability + filter (shared DDF/wide post-merge).

    mean_ia_prob stays ML-only (never polluted by the heuristic proxy). The
    coalescing chain for the filter is:

      1. mean_ia_prob      — cross-broker pooled ML P(Ia)          -> 'ml'
      2. sn_score          — single-classifier ML score (Fink SN-vs-other, or
                             the ALeRCE class probability for non-Ia classes,
                             whose Ia prob is deliberately NaN). Still ML,
                             class-level, so a ZTF SNII survives the filter
                             instead of being wrongly dropped        -> 'ml'
      3. antares_proxy_prob — capped heuristic; the object survives,
                             down-ranked, flagged needs_classification so the
                             observer knows a spectrum is a typing
                             observation, not a confirmed-Ia follow-up
                                                          -> 'antares_proxy'

    Critical during the Rubin downtime when ZTF-fed brokers are primary.
    Returns the frame filtered to effective_prob >= min_prob, with
    effective_prob / prob_source / needs_classification columns attached.
    """
    if 'mean_ia_prob' not in screened.columns:
        return screened
    ml = pd.to_numeric(screened['mean_ia_prob'], errors='coerce')
    score = pd.to_numeric(
        screened.get('sn_score', pd.Series(np.nan, index=screened.index)),
        errors='coerce')
    proxy = pd.to_numeric(
        screened.get('antares_proxy_prob',
                     pd.Series(np.nan, index=screened.index)),
        errors='coerce')
    screened = screened.copy()
    screened['effective_prob'] = ml.where(
        ml.notna(), score.where(score.notna(), proxy))
    screened['prob_source'] = np.where(
        ml.notna() | score.notna(), 'ml',
        np.where(proxy.notna(), 'antares_proxy', 'none'))
    screened['needs_classification'] = screened['prob_source'] != 'ml'
    before = len(screened)
    screened = screened[screened['effective_prob'] >= min_prob].copy()
    n_proxy = int((screened['prob_source'] == 'antares_proxy').sum())
    logger.info("After effective P(Ia) >= %.2f: %d (dropped %d; "
                "%d kept on ANTARES proxy, flagged needs_classification)",
                min_prob, len(screened), before - len(screened), n_proxy)
    return screened


def _normalize_merged_candidates(screened):
    """Normalize aggregator output columns for downstream compatibility.

    Shared by the DDF and wide paths. Ensures diaObjectId, sn_score and
    ddf_field exist — the merged row is built from scratch by the aggregator,
    so broker-specific ID/field columns must be coalesced back into the
    pipeline-wide schema.
    """
    if len(screened) == 0:
        return screened

    if 'diaObjectId' not in screened.columns:
        # Build diaObjectId from available ID columns, preferring Rubin IDs
        def _get_best_id(row):
            # Priority: rubin_dia_object_id > object_id_ANTARES > object_id > unique_id > coord-based
            for col in ['rubin_dia_object_id', 'object_id_ANTARES', 'object_id_Fink',
                        'object_id_ALeRCE', 'object_id_ALeRCE-ZTF', 'object_id',
                        'unique_id']:
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

    # ddf_field: the aggregator only carries it for brokers that set it
    # (ANTARES/ALeRCE blocks), so fill any missing rows from coordinates.
    # The column must be object-dtype first: an all-NaN merged column comes
    # out float64, and assigning field-name strings into it raises under
    # pandas>=3 (no silent object upcast).
    if 'ddf_field' not in screened.columns:
        screened['ddf_field'] = None
    screened['ddf_field'] = screened['ddf_field'].astype(object)
    missing_field = screened['ddf_field'].isna()
    if missing_field.any():
        screened.loc[missing_field, 'ddf_field'] = screened.loc[missing_field].apply(
            lambda r: is_in_ddf(r['ra'], r['dec']) if pd.notna(r.get('ra')) else None,
            axis=1,
        )

    return screened


def fetch_all_broker_candidates(fink, min_prob=0.3, days_back=30, n_fetch=500,
                                sky_mode='ddf', mjd_now=None, wide_kwargs=None,
                                fink_only=False):
    """Query all brokers for SN candidates, merge, deduplicate, screen variables.

    Brokers queried:
      - Fink LSST (always)
      - ALeRCE-ZTF (if available)
      - ALeRCE-LSST (if available)
      - ANTARES (if available)

    Returns a 2-tuple ``(candidates, broker_status)``:
      - candidates: merged, deduplicated DataFrame with columns
        diaObjectId, ra, dec, ddf_field, sn_score, brokers_detected,
        num_brokers, mean_ia_prob, known_variable, ...
      - broker_status: per-broker liveness dict, keyed by broker name, each
        ``{queried, responded, n_returned, error}``. Lets a silent broker
        outage be distinguished from a genuinely empty sky.
    """
    # --- Query Fink (if available) ---
    if fink is not None:
        fink_df = fetch_fink_candidates(fink, min_sn_score=min_prob,
                                        n_fetch=n_fetch, sky_mode=sky_mode)
        if sky_mode == 'wide' and len(fink_df) > 0:
            # Payload-level target-space selection BEFORE any per-object work
            fink_df = select_wide_candidates(fink_df, mjd_now,
                                             days_back=days_back,
                                             **(wide_kwargs or {}))
    else:
        logger.info("Fink API unavailable — skipping Fink candidate discovery")
        fink_df = pd.DataFrame()

    # --- Wide mode: Fink + live ALeRCE-ZTF through the real aggregator ---
    # ANTARES and ALeRCE-LSST are still DDF-restricted, so they are not
    # queried here; ZTF is queried directly with the same wide cuts. Since
    # phase 1 the aggregator carries the payload selection columns
    # (PASSTHROUGH_COLUMNS), so the cross-broker merge — coordinate dedup
    # with per-broker bookkeeping (num_brokers, brokers_detected,
    # object_id_<broker>) and cross-survey agreement stats — replaces the
    # old lightweight merge_wide_streams.
    if sky_mode == 'wide':
        wk = wide_kwargs or {}
        ztf_df, ztf_status = fetch_ztf_wide_candidates(
            mjd_now, min_prob=min_prob, days_back=days_back,
            dec_limit=wk.get('dec_limit', WIDE_DEC_LIMIT),
            max_mag=wk.get('max_mag', WIDE_MAX_MAG))
        # Fink-ZTF: live, out-of-region ZTF ingress. During the 2026-08 Chilean
        # storm this is the source that actually returns data (Rubin/Fink-LSST
        # dark, ALeRCE stalled); it merges co-equally as a third stream. Pure
        # top-of-funnel addition — identical schema, identical downstream.
        finkztf_df, finkztf_status = fetch_finkztf_wide_candidates(
            mjd_now, min_prob=min_prob, days_back=days_back,
            dec_limit=wk.get('dec_limit', WIDE_DEC_LIMIT),
            max_mag=wk.get('max_mag', WIDE_MAX_MAG))

        status = {
            'Fink': {'queried': fink is not None,
                     'responded': fink is not None and len(fink_df) >= 0,
                     'n_returned': int(len(fink_df)),
                     'error': None if fink is not None
                              else 'Fink client unavailable'},
            'ALeRCE-ZTF': ztf_status,
            'Fink-ZTF': finkztf_status,
            'ANTARES': {'queried': False, 'responded': False,
                        'n_returned': 0,
                        'error': 'not queried (wide mode; DDF-cone only)'},
            'ALeRCE-LSST': {'queried': False, 'responded': False,
                            'n_returned': 0,
                            'error': 'not queried (wide mode; DDF-cone only)'},
        }

        # Cross-broker merge. 2" tolerance matches the old wide merge;
        # extinction lookup off (network) — wide merit treats E_BV as neutral.
        # Local import: the aggregator is otherwise only reached through the
        # optional SupernovaMonitor, and DDF-only runs must not require it.
        from core.alert_aggregator import AlertAggregator
        aggregator = AlertAggregator(cache_dir='./cache/data',
                                     match_tolerance_arcsec=2.0,
                                     apply_extinction=False)
        combined = aggregator.merge_alerts({'Fink': fink_df,
                                            'ALeRCE-ZTF': ztf_df,
                                            'Fink-ZTF': finkztf_df})
        logger.info("Wide merge: %d Fink + %d ALeRCE-ZTF + %d Fink-ZTF -> %d unique",
                    len(fink_df), len(ztf_df), len(finkztf_df), len(combined))
        if len(combined) == 0:
            return pd.DataFrame(), status

        # NOTE: no monitor.variable_screener here — it needs SupernovaMonitor
        # and its catalogs only cover the DDFs. Wide mode's variable screens
        # are the payload cross-matches (select_wide_candidates) and the
        # SQL-level baseline cut in fetch_ztf_wide_candidates.
        combined = _coalesce_effective_prob(combined, min_prob)
        combined = _normalize_merged_candidates(combined)
        return combined, status

    # The non-Fink brokers we *intend* to query in multi-broker mode. Used to
    # build a fallback broker-status block if the multi-broker path fails so a
    # silent broker outage is never mistaken for an empty sky (critical during
    # the Rubin offline window when ZTF-fed brokers are the primary source).
    _OTHER_BROKERS = ('ANTARES', 'ALeRCE-ZTF', 'ALeRCE-LSST')

    def _fink_only_status(fink_n, reason=None):
        status = {
            'Fink': {
                'queried': True,
                'responded': fink is not None,
                'n_returned': int(fink_n),
                'error': None if fink is not None else 'Fink client unavailable',
            }
        }
        for b in _OTHER_BROKERS:
            status[b] = {
                'queried': True,
                'responded': False,
                'n_returned': 0,
                'error': reason or 'not queried (Fink-only mode)',
            }
        return status

    if fink_only or not HAS_MONITOR:
        if not HAS_MONITOR and not fink_only:
            logger.warning("SupernovaMonitor not available; using Fink only")
        # Fink-only path: just assign fields and return
        reason = ('SupernovaMonitor not available' if not HAS_MONITOR
                  else 'not queried (Fink-only mode)')
        if len(fink_df) == 0:
            return pd.DataFrame(), _fink_only_status(0, reason)
        fink_df['brokers_detected'] = 'Fink'
        fink_df['num_brokers'] = 1
        fink_df['mean_ia_prob'] = fink_df['sn_score']
        fink_df['known_variable'] = False
        return fink_df, _fink_only_status(len(fink_df), reason)

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
        broker_status = dict(getattr(monitor, '_last_broker_status', {}) or {})
    except Exception as e:
        # Multi-broker path blew up wholesale: record every non-Fink broker as
        # queried-but-unresponsive so the failure is visible in the report.
        logger.warning("Multi-broker query failed: %s. Using Fink only.", e)
        status = _fink_only_status(len(fink_df), reason=str(e))
        status['Fink']['responded'] = fink is not None
        if len(fink_df) == 0:
            return pd.DataFrame(), status
        fink_df['brokers_detected'] = 'Fink'
        fink_df['num_brokers'] = 1
        fink_df['mean_ia_prob'] = fink_df['sn_score']
        fink_df['known_variable'] = False
        return fink_df, status

    # Log per-broker counts
    for broker_name, bdf in other_brokers.items():
        n = len(bdf) if bdf is not None else 0
        logger.info("  %s: %d candidates", broker_name, n)

    # --- Record Fink's own liveness (queried outside query_all_brokers) ---
    broker_status['Fink'] = {
        'queried': True,
        'responded': fink is not None,
        'n_returned': int(len(fink_df)),
        'error': None if fink is not None else 'Fink client unavailable',
    }

    # --- Add Fink to the broker dict ---
    if len(fink_df) > 0:
        other_brokers['Fink'] = fink_df

    # --- Merge and deduplicate across all brokers ---
    aggregator = monitor.aggregator
    merged = aggregator.merge_alerts(other_brokers)

    if len(merged) == 0:
        logger.warning("No candidates after merge")
        return pd.DataFrame(), broker_status

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

    # --- Coalesced effective probability + filter (shared with wide mode) ---
    screened = _coalesce_effective_prob(screened, min_prob)

    # --- Normalize output columns for downstream compatibility ---
    # Need: diaObjectId, ra, dec, ddf_field, sn_score (shared with wide mode)
    screened = _normalize_merged_candidates(screened)

    logger.info("Final candidates: %d from %s",
                len(screened),
                ', '.join(sorted(set(screened.get('brokers_detected', pd.Series(['Fink'])).dropna()))))

    return screened, broker_status


ATLAS_BRIGHT_MAG_CUT = 20.0  # Only fetch ATLAS for candidates brighter than this

# Allocations file for the LLAMAS orchestrator (the single scheduling
# authority; the pipeline itself only ranks). The example file carries
# ILLUSTRATIVE budgets until MAGNETS agrees real per-PI numbers.
DEFAULT_ALLOCATIONS = 'ref/allocations_example.yaml'

# Reference exposure for the merit-per-hour ranking (matches the scheduler's
# value-density reference): a 45-min target has density 1.0.
RATE_REF_MINUTES = 45.0


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


def resolve_salt_mode(sky_mode, use_salt_flag, no_salt_flag,
                      has_sncosmo=HAS_SNCOSMO):
    """SALT2 default policy (pure, testable).

    SALT is ON by default in wide sky mode (typing matters most there),
    opt-in via --use-salt in ddf mode, and --no-salt always wins.
    Requires sncosmo regardless.
    """
    want = (use_salt_flag or sky_mode == 'wide') and not no_salt_flag
    return bool(want and has_sncosmo)


def fetch_and_fit(fink, candidates_df, mjd_now, fetch_ztf=True, fetch_atlas=True,
                  min_snr_points=5, min_bands=2, min_fit_bands=2,
                  prefilter_min_sources=0, use_salt=False, redshifts=None,
                  max_rise_time=30.0, max_phase_days=25.0,
                  max_baseline_days=150.0, ebv_lookup=None,
                  salt_rescue_cap=30):
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
      redshifts: Redshift info for SALT fitting keyed by diaObjectId. New style:
                 {did: {'z': float-or-nan, 'source': 'tns_specz'|'legacy_photz'|'none'}}.
                 Old style {did: float} is still accepted (treated as an
                 external spec-z, i.e. fixed in the fit).
      ebv_lookup: Optional {did: E(B-V)} for the SALT MW-dust component.
      salt_rescue_cap: Max number of SALT "rescue" fits per run — objects that
                 FAILED the generic Villar/parabola gate still get a SALT
                 attempt (in encounter order) up to this cap; a good template
                 fit rescues them into the results (salt_rescued=True).
      max_rise_time: Maximum rise time in days (default 30). SNe Ia rise in
                 ~17-20d. Not applied to SALT-anchored fits (the template
                 already enforces an Ia rise).
    """
    # Normalize redshift info to {did: {'z': ..., 'source': ...}}
    z_info = {}
    for _did, _v in (redshifts or {}).items():
        if isinstance(_v, dict):
            z_info[_did] = {'z': _v.get('z', np.nan),
                            'source': _v.get('source', 'none')}
        else:
            try:
                _zval = float(_v)
            except (TypeError, ValueError):
                _zval = np.nan
            # Backward compat: a bare float was historically a NED host
            # spec-z passed as a fixed z — keep that behavior.
            z_info[_did] = {'z': _zval, 'source': 'tns_specz'}
    ebv_lookup = ebv_lookup or {}
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
    # Breaker thresholds come from core.fink_breaker (FINK_MAX_CONSECUTIVE_FAILURES, ...).
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
        # Pause-and-resume circuit breaker: on a run of consecutive *transport*
        # errors we cool down (sleep) and reset rather than aborting the night.
        # An empty light curve (object queried OK, no photometry) is NOT a
        # failure — it's just skipped. The breaker decision logic lives in the
        # pure, testable FinkBreaker helper; the sleep stays here.
        breaker = FinkBreaker()
        fink_empty = 0
        for i, did in enumerate(dia_ids):
            action = breaker.decide()
            if action == ACTION_COOLDOWN:
                logger.warning("Fink: %d consecutive transport failures — "
                               "cooling down %ds (cooldown %d/%d), then resuming",
                               breaker.consecutive_failures,
                               FINK_COOLDOWN_SECONDS,
                               breaker.cooldowns_used + 1, FINK_MAX_COOLDOWNS)
                time.sleep(FINK_COOLDOWN_SECONDS)
                breaker.record_cooldown()
            elif action == ACTION_PROCEED:
                # Cooldown budget exhausted: stop sleeping but keep going.
                logger.warning("Fink: cooldown budget exhausted (%d) — "
                               "continuing without further pauses",
                               FINK_MAX_COOLDOWNS)
                breaker.record_cooldown()  # resets streak so we don't re-trip every row

            logger.info("[%d/%d] Fink: %s", i + 1, len(dia_ids), did)
            fink_lc = fink.get_light_curve(str(did), include_forced=True)

            if fink_lc is None:
                # Transport error — counts toward the breaker.
                breaker.record_failure()
                logger.warning("  Fink transport error (consecutive: %d/%d)",
                               breaker.consecutive_failures,
                               FINK_MAX_CONSECUTIVE_FAILURES)
                continue
            if len(fink_lc) == 0:
                # Queried OK, no photometry — NOT a failure; just skip.
                breaker.record_success()
                fink_empty += 1
                continue

            # Success — reset the consecutive-failure streak.
            breaker.record_success()
            fink_data[did] = fink_lc

            # Check if any Rubin detection is brighter than the ATLAS cut
            if fetch_atlas and 'magnitude' in fink_lc.columns:
                mags = pd.to_numeric(fink_lc['magnitude'], errors='coerce')
                brightest = mags.dropna().min()
                if np.isfinite(brightest) and brightest < ATLAS_BRIGHT_MAG_CUT:
                    ra, dec = coord_lookup.get(did, (np.nan, np.nan))
                    if np.isfinite(ra) and np.isfinite(dec):
                        bright_for_atlas.append((str(did), ra, dec))

        logger.info("Fink photometry: %d/%d candidates have data "
                    "(%d empty-OK, %d cooldowns)",
                    len(fink_data), len(dia_ids), fink_empty,
                    breaker.cooldowns_used)

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
    n_salt_rescues = 0  # tier-2 SALT attempts spent (capped at salt_rescue_cap)
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

        # Long-baseline check: a SN's significant detections span weeks-months;
        # a source detected across >max_baseline_days is a long-lived variable
        # (AGN/QSO) — reject BEFORE spending a fit on it. This generalizes what
        # the ZTF cross-match catches incidentally for the few objects with
        # archival ZTF data, and stands in for the DDF-only variable catalogs
        # in wide sky mode. Uses high-SNR points so a long forced-photometry
        # (non-detection) baseline does not trigger it.
        det_span = (high_snr['mjd'].max() - high_snr['mjd'].min()
                    if len(high_snr) > 1 else 0.0)
        if det_span > max_baseline_days:
            logger.warning("  Long-lived source (%.0fd detection baseline > %.0fd)"
                           " — likely AGN/variable, skipping", det_span,
                           max_baseline_days)
            continue

        logger.info("  %d pts (%d SNR>5) in %d bands (%.0fd span): %s",
                    len(lc_clean), n_high_snr, n_bands_detected, mjd_span,
                    ', '.join(f"{b}={n}" for b, n in band_counts.items()))

        # --- Run multiband Villar fit (primary) and parabola (fallback) ---
        vil = fit_villar_multiband(combined)
        par = fit_parabola(combined)

        mjd_min = float(lc_clean['mjd'].min())
        mjd_max = float(lc_clean['mjd'].max())

        # Would Villar/parabola alone pass the generic gate? Decides whether
        # a SALT fit is tier 1 (cross-check on a passing object) or tier 2
        # (a "rescue" attempt on a failing one, subject to salt_rescue_cap).
        _, generic_method = choose_best_fit(vil, par, None, min_fit_bands,
                                            mjd_min, mjd_max)

        # --- SALT2 template fit (tier 1 always; tier 2 up to the cap) ---
        salt_result = None
        if use_salt and HAS_SNCOSMO:
            is_rescue = (generic_method == 'none')
            if is_rescue and n_salt_rescues >= salt_rescue_cap:
                logger.debug("  SALT rescue cap (%d) reached — not attempting",
                             salt_rescue_cap)
            else:
                if is_rescue:
                    n_salt_rescues += 1
                zi = z_info.get(did) or {}
                z_fixed, z_bounds = salt_z_policy(zi.get('z', np.nan),
                                                  zi.get('source'))
                salt_result = fit_salt(combined, model_name='salt2-extended',
                                       z=z_fixed, z_bounds=z_bounds,
                                       mwebv=ebv_lookup.get(did))
                if salt_result.get('status') == 'ok':
                    logger.info("  SALT2: x1=%.2f, c=%.2f, chi2/dof=%.1f, "
                                "z=%.3f%s, t0_err=%.1fd",
                                salt_result.get('x1', np.nan),
                                salt_result.get('c', np.nan),
                                salt_result.get('chi2_dof', np.nan),
                                salt_result.get('z', np.nan),
                                ' (railed)' if salt_result.get('z_railed')
                                else '',
                                salt_result.get('t0_err', np.nan))

        # --- Final decision: SALT (if trustworthy) > Villar > parabola ---
        best, fit_method = choose_best_fit(vil, par, salt_result,
                                           min_fit_bands, mjd_min, mjd_max)
        salt_rescued = (fit_method == 'salt' and generic_method == 'none')
        if best is None:
            vil_best = vil.get('best')
            par_bands_ok = sum(1 for info in par.get('per_band', {}).values()
                               if info.get('status') == 'ok')
            logger.warning("  No acceptable fit (Villar %s in %d bands, "
                           "parabola %d bands ok, SALT %s) — skipping",
                           vil_best.get('status', 'none') if vil_best else 'none',
                           vil.get('n_bands_fit', 0), par_bands_ok,
                           salt_result.get('status', 'not_run')
                           if salt_result else 'not_run')
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
        # Phase gate matched to the merit timescale (w_time, tau=10d): beyond
        # ~2.5*tau the merit is <~0.05 and the target is spectroscopically
        # stale anyway. The old +/-60d gate admitted targets whose merit then
        # rounded to 0.000, filling the plan with unrankable objects.
        if abs(delta_t) > max_phase_days:
            logger.warning("  Peak too far from now (dt=%.0fd, limit %.0fd) — skipping",
                           delta_t, max_phase_days)
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

        # Rise time filter: SNe Ia rise in ~17-20 days, reject slow risers.
        # Skipped for SALT-anchored fits — the template already enforces an
        # Ia rise, and the first-detection proxy is meaningless there.
        MIN_RISE_TIME = 5.0   # days (reject if peak is before first detection)
        if fit_method != 'salt' and np.isfinite(rise_time):
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
            'salt_rescued': salt_rescued,
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
        elif pd.notna(mean_prob):
            ia_prob = mean_prob
        else:
            # Coalesced fallback (ANTARES-only objects): the capped heuristic
            # proxy, so the object ranks low instead of vanishing. Flagged
            # needs_classification downstream.
            ia_prob = cand.get('effective_prob', np.nan)

        # Ia-SPECIFIC evidence (distinct from generic SN-vs-other prob):
        # TNS spectroscopic Ia is definitive; ALeRCE lc_classifier gives a
        # per-class prob (SNIa positive, other SN classes explicit non-Ia);
        # Fink earlySNIa applies to young objects. NaN = no info = neutral.
        # Evidence 0.5 maps to the neutral factor (w_iaspec = 1.0), so any
        # POSITIVE Ia classification is placed in [0.5, 1] — a predicted Ia
        # must never rank below an unclassified object, however weak the
        # classifier confidence (PI requirement: prefer Ia where possible).
        ia_evidence = np.nan
        tns_type = str(cand.get('tns_type') or '')
        alerce_cls = str(cand.get('alerce_class') or '')
        early_ia = pd.to_numeric(pd.Series([cand.get('early_ia_score')]),
                                 errors='coerce').iloc[0]
        if tns_type.startswith('SN Ia'):
            ia_evidence = 1.0
        elif alerce_cls == 'SNIa':
            p = float(pd.to_numeric(
                pd.Series([cand.get('mean_ia_prob')]), errors='coerce'
            ).fillna(0.5).iloc[0])
            ia_evidence = 0.5 + 0.5 * min(max(p, 0.0), 1.0)
        elif alerce_cls in ('SNII', 'SNIbc', 'SLSN'):
            ia_evidence = 0.0  # positively classified non-Ia -> mild demotion
        if pd.notna(early_ia) and early_ia > 0:
            ia_evidence = np.nanmax([ia_evidence,
                                     0.5 + 0.5 * min(float(early_ia), 1.0)])

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
        # Coverage-aware broker bonus: southern (dec <= -32) DDFs can only ever
        # be seen by LSST-fed brokers, so cap the achievable broker count by dec.
        max_brokers = max_possible_brokers(cand['dec'])

        # Get redshift info
        z_info = redshifts.get(did, {})
        redshift = z_info.get('redshift', np.nan) if z_info else np.nan
        distmod = z_info.get('distmod', np.nan) if z_info else np.nan
        ned_name = z_info.get('ned_name', '') if z_info else ''
        ned_sep = z_info.get('separation_arcsec', np.nan) if z_info else np.nan

        # Get SALT fit results
        salt = fit.get('salt')
        salt_status = salt.get('status', '') if salt else ''
        salt_ok = bool(salt) and salt.get('status') == 'ok'
        salt_x1 = salt.get('x1', np.nan) if salt_ok else np.nan
        salt_c = salt.get('c', np.nan) if salt_ok else np.nan
        salt_chi2_dof = salt.get('chi2_dof', np.nan) if salt_ok else np.nan
        salt_z = salt.get('z', np.nan) if salt_ok else np.nan
        salt_peak_mag_B = salt.get('peak_mag_B', np.nan) if salt_ok else np.nan
        salt_t0 = salt.get('t0', np.nan) if salt_ok else np.nan
        salt_t0_err = salt.get('t0_err', np.nan) if salt_ok else np.nan
        salt_z_railed = bool(salt.get('z_railed', False)) if salt_ok else False
        salt_rescued = bool(fit.get('salt_rescued', False))

        # Compute absolute magnitude if we have redshift. Prefer the SALT
        # rest-frame B peak — but ONLY when the redshift came from an
        # external source (TNS spec-z / Legacy photo-z). When SALT floated
        # z freely, M_B ≈ −19.4 is baked into the template–distance
        # degeneracy, so using it would be circular.
        cand_z_source = str(cand.get('z_source', '') or '')
        absolute_mag = np.nan
        if (salt_ok and np.isfinite(salt_peak_mag_B) and np.isfinite(distmod)
                and cand_z_source in ('tns_specz', 'legacy_photz')):
            absolute_mag = salt_peak_mag_B - distmod
        elif np.isfinite(peak_mag) and np.isfinite(distmod):
            absolute_mag = peak_mag - distmod

        # Merit score with all factors. Moon penalty is NOT applied here — it
        # depends on the observing night and is folded in after
        # filter_observable_targets() via recompute_merit_with_moon(), which
        # re-sorts the plan. Use the breakdown to get individual component weights.
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
                max_possible_brokers=max_brokers,
                salt_chi2_dof=salt_arg,
                absolute_mag=absmag_arg,
                ia_evidence=ia_evidence,
            )
            # Per-program ranking: the same candidate scored under the
            # exotic-transients profile (rising-preferring, Ia-evidence
            # avoided, Ia-specific factors off). Programs rank the SAME
            # list differently; see RankingProfile.
            breakdown_x = compute_merit_breakdown(
                delta_t, peak_mag,
                ia_prob=prob_arg,
                host_morphology=host_morph,
                extinction_ebv=ext_arg,
                num_brokers=num_brokers,
                max_possible_brokers=max_brokers,
                salt_chi2_dof=salt_arg,
                absolute_mag=absmag_arg,
                ia_evidence=ia_evidence,
                profile=EXOTIC_PROFILE,
            )
            merit = float(breakdown['merit'])
            w_time = float(breakdown['w_time'])
            w_mag = float(breakdown['w_mag'])
            w_prob = float(breakdown['w_prob'])
            w_host = float(breakdown['w_host'])
            w_ext = float(breakdown['w_ext'])
            w_broker = float(breakdown['w_broker'])
            w_moon = float(breakdown['w_moon'])
            w_salt = float(breakdown['w_salt'])
            w_absmag = float(breakdown['w_absmag'])
        else:
            merit = np.nan
            w_time = w_mag = w_prob = w_host = w_ext = w_broker = np.nan
            w_moon = np.nan
            w_salt = w_absmag = np.nan

        rows.append({
            'diaObjectId': did,
            'ra': cand['ra'],
            'dec': cand['dec'],
            'ddf_field': cand.get('ddf_field', ''),
            'sn_score': cand.get('sn_score', np.nan),
            'early_ia_score': cand.get('early_ia_score', np.nan),
            'alerce_class': cand.get('alerce_class', ''),
            'alerce_classifier': cand.get('alerce_classifier', ''),
            'z_source': cand.get('z_source', ''),
            'ztf_oid': cand.get('ztf_oid', ''),
            'brokers_detected': cand.get('brokers_detected', 'Fink'),
            'num_brokers': num_brokers,
            'max_possible_brokers': max_brokers,
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
            'salt_t0': salt_t0,
            'salt_t0_err': salt_t0_err,
            'salt_z_railed': salt_z_railed,
            'salt_rescued': salt_rescued,
            # Merit breakdown
            'merit': merit,
            'w_time': w_time,
            'w_mag': w_mag,
            'w_prob': w_prob,
            'w_host': w_host,
            'w_ext': w_ext,
            'w_broker': w_broker,
            # w_moon / moon_penalty start neutral; recompute_merit_with_moon()
            # overwrites them (and re-sorts merit) once the night's moon is known.
            'w_moon': w_moon,
            'moon_penalty': float('nan'),
            'w_salt': w_salt,
            'w_absmag': w_absmag,
            'w_iaspec': float(breakdown['w_iaspec']),
            'ia_evidence': ia_evidence,
            'prob_source': cand.get('prob_source', 'ml'),
            'needs_classification': bool(cand.get('needs_classification', False)),
            # Same candidate under the exotic-transients ranking profile
            'merit_exotic': float(breakdown_x['merit']),
            'object_id': did,  # alias for magellan_planning
        })

    summary = pd.DataFrame(rows)
    if len(summary) > 0:
        summary = summary.sort_values('merit', ascending=False, na_position='last')
    return summary


def enrich_finalist_redshifts(summary, fit_results, use_salt=False):
    """Fill in redshifts for FINAL candidates that lack one (TNS then NED).

    Runs after ranking, on the ~30 finalists only — unlike the retired
    every-candidate TNS crawl (320 rate-limited queries), this is a handful
    of lookups for objects we might actually observe. Matters mostly for the
    ZTF stream, whose payload carries no redshift: a gained z fixes the
    exposure estimate (redshift table instead of magnitude scaling) and
    enables the absolute-magnitude consistency check.

    Precedence: TNS spectroscopic z > NED host z. When a z is gained and the
    object has a good SALT fit, the SALT model is refit with the redshift
    FIXED (typing/w_salt/abs-mag update only — the fitted peak/phase and the
    ranking gates are deliberately left untouched to avoid re-gating loops).
    """
    if len(summary) == 0 or 'redshift' not in summary.columns:
        return summary
    # String columns we may write into can arrive all-NaN (float64); pandas>=3
    # refuses silent object upcasts on assignment.
    for col in ('tns_name', 'tns_type', 'ned_name'):
        if col in summary.columns:
            summary[col] = summary[col].astype(object)
    zless = summary[~(pd.to_numeric(summary['redshift'], errors='coerce') > 0)]
    if len(zless) == 0:
        return summary
    logger.info("Redshift enrichment: %d/%d finalists lack z — querying TNS + NED...",
                len(zless), len(summary))

    from astropy.cosmology import FlatLambdaCDM
    _cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

    # --- TNS: local cross-match against the cached daily dump (ONE download
    # per night instead of per-object rate-limited cone searches); serial
    # cone-search fallback only if the dump is unavailable. ---
    n_tns = n_ned = 0
    gained = {}  # idx -> (z, source_label)
    tns_hits = {}
    if HAS_TNS:
        try:
            from broker_clients.tns_client import (fetch_tns_public_objects,
                                                   crossmatch_tns_local)
            dump = fetch_tns_public_objects()
            if dump is not None:
                tns_hits = crossmatch_tns_local(
                    zless[['diaObjectId', 'ra', 'dec']], dump)
        except Exception as e:
            logger.warning("z-enrichment: TNS dump path failed: %s", e)

    if tns_hits:
        for idx, row in zless.iterrows():
            hit = tns_hits.get(str(row['diaObjectId']))
            if not hit:
                continue
            summary.at[idx, 'tns_name'] = hit['tns_name']
            if hit['tns_type']:
                summary.at[idx, 'tns_type'] = hit['tns_type']
            summary.at[idx, 'tns_match'] = True
            z = hit['tns_redshift']
            if z is not None and z > 0:
                summary.at[idx, 'tns_redshift'] = z
                gained[idx] = (z, 'tns_specz')
                n_tns += 1
    elif HAS_TNS:
        # Fallback: serial cone searches (rate-limited; finalists only)
        tns = None
        try:
            tns = TNSClient()
            ok, msg = tns.verify_connection()
            if not ok:
                logger.warning("z-enrichment: TNS unavailable (%s)", msg)
                tns = None
        except Exception as e:
            logger.warning("z-enrichment: TNS init failed: %s", e)
            tns = None
        for idx, row in zless.iterrows():
            if tns is None:
                break
            try:
                matches = tns.search_by_coordinates(row['ra'], row['dec'],
                                                    radius_arcsec=5.0)
            except Exception as e:
                logger.warning("z-enrichment: TNS query failed (%s) — stopping TNS pass", e)
                break
            for m in (matches or []):
                z = m.get('redshift')
                if z is not None and np.isfinite(float(z)) and float(z) > 0:
                    gained[idx] = (float(z), 'tns_specz')
                    name = f"{m.get('prefix', 'AT')} {m.get('objname', '')}".strip()
                    summary.at[idx, 'tns_name'] = name
                    if m.get('type'):
                        summary.at[idx, 'tns_type'] = m['type']
                    summary.at[idx, 'tns_redshift'] = float(z)
                    summary.at[idx, 'tns_match'] = True
                    n_tns += 1
                    break

    # --- NED host z for the remainder ---
    still = zless.index.difference(gained.keys())
    if HAS_NED and len(still) > 0:
        try:
            from cache.alert_cache import AlertCache
            ned_df = query_ned_batch(
                summary.loc[still, ['diaObjectId', 'ra', 'dec']].copy(),
                cache=AlertCache(), radius_arcsec=18.0)
            for _, nrow in ned_df.iterrows():
                z = nrow.get('ned_redshift')
                if pd.notna(z) and z > 0:
                    match = summary.index[summary['diaObjectId'] == nrow['diaObjectId']]
                    if len(match):
                        gained[match[0]] = (float(z), f"ned:{nrow.get('ned_name', '')}")
                        n_ned += 1
        except Exception as e:
            logger.warning("z-enrichment: NED pass failed: %s", e)

    # --- Apply: redshift, distmod, absolute mag, optional fixed-z SALT refit ---
    n_refit = 0
    for idx, (z, source) in gained.items():
        summary.at[idx, 'redshift'] = z
        distmod = float(_cosmo.distmod(z).value)
        summary.at[idx, 'distmod'] = distmod
        summary.at[idx, 'ned_name'] = source
        peak_mag = summary.at[idx, 'peak_mag']
        if np.isfinite(peak_mag):
            summary.at[idx, 'absolute_mag'] = float(peak_mag) - distmod
        if use_salt and HAS_SNCOSMO:
            did = summary.at[idx, 'diaObjectId']
            fit = fit_results.get(did, {})
            lc = fit.get('light_curve_clean')
            if lc is not None and len(lc) > 0:
                try:
                    refit = fit_salt(lc, z=z)
                    if refit.get('status') == 'ok':
                        summary.at[idx, 'salt_chi2_dof'] = refit.get('chi2_dof', np.nan)
                        summary.at[idx, 'salt_x1'] = refit.get('x1', np.nan)
                        summary.at[idx, 'salt_c'] = refit.get('c', np.nan)
                        summary.at[idx, 'salt_z'] = z
                        pmB = refit.get('peak_mag_B', np.nan)
                        if np.isfinite(pmB):
                            summary.at[idx, 'salt_peak_mag_B'] = pmB
                            summary.at[idx, 'absolute_mag'] = float(pmB) - distmod
                        n_refit += 1
                except Exception as e:
                    logger.debug("z-enrichment: SALT refit failed for %s: %s", did, e)

    logger.info("Redshift enrichment: gained %d (TNS spec-z %d, NED host %d); "
                "%d SALT fixed-z refits", len(gained), n_tns, n_ned, n_refit)
    return summary


def enrich_finalist_typing(summary, fit_results):
    """Multi-type template tournament over the finalists (typing evidence).

    For each finalist, the SALT2 (Ia) fit competes against the core-collapse
    template set (nugent Ibc/IIP/IIn) on the same cleaned light curve under
    the same redshift policy. Positive typing evidence instead of the
    one-sided "not a good Ia" signal (x1 rail / bad chi2): the winner says
    WHICH template the photometry prefers. Ground truth 2026-07-13:
    6/6 classified finalists consistent (Ia->salt2, SN II->IIP, TDE->IIn).

    Adds columns:
      template_best       winning label ('Ia', 'Ibc', 'IIP', 'IIn')
      template_best_chi2  winner's chi2/dof
      template_margin     runner-up chi2/dof - winner's (decisiveness)
      template_peak_mjd   winner's synthesized peak epoch (non-Ia winners)

    Evidence only: merit, phase and ranking are deliberately untouched —
    non-Ia phase correction and an exotic rescue tier are the fit-loop
    follow-on. Cost: ~3 extra ~1 s fits x ~30 finalists.
    """
    if len(summary) == 0 or not HAS_SNCOSMO:
        return summary
    from core.peak_fitting import run_template_tournament

    summary['template_best'] = pd.Series(index=summary.index, dtype=object)
    for col in ('template_best_chi2', 'template_margin', 'template_peak_mjd'):
        summary[col] = np.nan

    n_run = 0
    wins = {}
    for idx, row in summary.iterrows():
        did = row['diaObjectId']
        fit = fit_results.get(did, {})
        lc = fit.get('light_curve_clean')
        if lc is None or len(lc) < 5:
            continue
        z_val = pd.to_numeric(row.get('redshift'), errors='coerce')
        src = row.get('z_source')
        if str(row.get('ned_name') or '') == 'tns_specz':
            src = 'tns_specz'   # z gained by enrichment: fix it
        z_fixed, z_bounds = salt_z_policy(z_val, src)
        salt = fit.get('salt')
        mwebv = (salt or {}).get('mwebv')
        try:
            t = run_template_tournament(lc, z=z_fixed, z_bounds=z_bounds,
                                        salt=salt, mwebv=mwebv, clean=False)
        except Exception as e:
            logger.debug("tournament failed for %s: %s", did, e)
            continue
        if t.get('status') != 'ok':
            continue
        n_run += 1
        best = t['template_best']
        wins[best] = wins.get(best, 0) + 1
        summary.at[idx, 'template_best'] = best
        summary.at[idx, 'template_best_chi2'] = t['template_best_chi2_dof']
        summary.at[idx, 'template_margin'] = t['template_margin']
        if best != 'Ia':
            summary.at[idx, 'template_peak_mjd'] = t['template_peak_mjd']

    non_ia = {k: v for k, v in wins.items() if k != 'Ia'}
    logger.info("Template tournament: %d/%d finalists fit; winners: %s%s",
                n_run, len(summary),
                ', '.join(f"{k}={v}" for k, v in sorted(wins.items())),
                (' — non-Ia preferred: check typing before long exposures'
                 if non_ia else ''))
    return summary


def recompute_merit_with_moon(plan_df):
    """Fold the night's moon penalty into the ranking merit and re-sort.

    build_summary_table() computes merit without a moon penalty (the night is
    not yet known there). filter_observable_targets() then attaches a per-target
    ``moon_penalty`` column. This recomputes merit + every ``w_*`` component
    (including ``w_moon``) using that real penalty, overwrites those columns in
    place, and re-sorts by the moon-aware merit so the moon actually drives the
    ranking. The report writer and candidates.csv both read these columns, so
    there is a single source of truth for merit and its breakdown.

    Parameters
    ----------
    plan_df : pd.DataFrame
        Output of filter_observable_targets(), must carry the columns produced
        by build_summary_table() plus ``moon_penalty``.

    Returns
    -------
    pd.DataFrame — copy with merit / w_* recomputed and re-sorted by merit.
    """
    if len(plan_df) == 0 or 'moon_penalty' not in plan_df.columns:
        return plan_df

    df = plan_df.copy()

    # Ensure the columns we overwrite are float-typed so in-place assignment
    # of recomputed weights never hits a dtype (int->float) cast error.
    merit_cols = ['merit', 'merit_exotic', 'w_time', 'w_mag', 'w_prob',
                  'w_host', 'w_ext', 'w_broker', 'w_moon', 'w_salt',
                  'w_absmag', 'w_iaspec']
    for col in merit_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    def _arg(row, col):
        val = row.get(col, np.nan)
        return val if (val is not None and np.isfinite(val)) else None

    for idx, row in df.iterrows():
        delta_t = row.get('delta_t', np.nan)
        peak_mag = row.get('peak_mag', np.nan)
        if not (np.isfinite(delta_t) and np.isfinite(peak_mag)):
            continue

        moon_pen = row.get('moon_penalty', np.nan)
        breakdown = compute_merit_breakdown(
            delta_t, peak_mag,
            ia_prob=_arg(row, 'sn_score') if (
                pd.notna(row.get('sn_score')) and float(row.get('sn_score', 0)) > 0
            ) else (_arg(row, 'mean_ia_prob')
                    if _arg(row, 'mean_ia_prob') is not None
                    else _arg(row, 'effective_prob')),
            host_morphology=row.get('host_morphology', 'unknown'),
            extinction_ebv=_arg(row, 'E_BV'),
            num_brokers=row.get('num_brokers', 1),
            max_possible_brokers=row.get('max_possible_brokers'),
            moon_penalty=moon_pen if np.isfinite(moon_pen) else None,
            salt_chi2_dof=_arg(row, 'salt_chi2_dof'),
            absolute_mag=_arg(row, 'absolute_mag'),
            ia_evidence=_arg(row, 'ia_evidence'),
        )
        df.at[idx, 'merit'] = float(breakdown['merit'])
        for key in ('w_time', 'w_mag', 'w_prob', 'w_host', 'w_ext',
                    'w_broker', 'w_moon', 'w_salt', 'w_absmag', 'w_iaspec'):
            if key in df.columns:
                df.at[idx, key] = float(breakdown[key])
        # Exotic-profile merit gets the same moon treatment
        if 'merit_exotic' in df.columns:
            breakdown_x = compute_merit_breakdown(
                delta_t, peak_mag,
                ia_prob=_arg(row, 'sn_score') if (
                    pd.notna(row.get('sn_score')) and float(row.get('sn_score', 0)) > 0
                ) else (_arg(row, 'mean_ia_prob')
                        if _arg(row, 'mean_ia_prob') is not None
                        else _arg(row, 'effective_prob')),
                host_morphology=row.get('host_morphology', 'unknown'),
                extinction_ebv=_arg(row, 'E_BV'),
                num_brokers=row.get('num_brokers', 1),
                max_possible_brokers=row.get('max_possible_brokers'),
                moon_penalty=moon_pen if np.isfinite(moon_pen) else None,
                salt_chi2_dof=_arg(row, 'salt_chi2_dof'),
                absolute_mag=_arg(row, 'absolute_mag'),
                ia_evidence=_arg(row, 'ia_evidence'),
                profile=EXOTIC_PROFILE,
            )
            df.at[idx, 'merit_exotic'] = float(breakdown_x['merit'])

    df = df.sort_values('merit', ascending=False, na_position='last')
    return df.reset_index(drop=True)


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
                        pdf_path, mjd_now, obs_date, observing_sequence=None,
                        broker_status=None, orch_dir=None):
    """Generate multi-page PDF report with summary and light curves.

    If ``orch_dir`` points at the orchestrator's output directory, the
    executable LLAMAS plan (timeline + per-program charges) is rendered as a
    page so the PDF is self-contained for observers and PIs.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    has_rate = ('merit_rate' in summary_df.columns
                and summary_df['merit_rate'].notna().any())

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

        if fields:
            stats = (f'{fields} DDFs  |  {n_atlas} with ATLAS  |  {n_ztf} with ZTF  |  '
                     f'{high_merit} high-merit (>0.1)')
        else:
            n_rubin = ((summary_df['n_ztf'] == 0).sum()
                       if 'n_ztf' in summary_df.columns else 0)
            stats = (f'wide-sky selection  |  {n_ztf} with ZTF photometry  |  '
                     f'{n_rubin} Rubin-only  |  {high_merit} high-merit (>0.1)')
        ax.text(0.5, 0.20, stats,
                ha='center', va='center', fontsize=10, color='dimgray')

        # Broker-liveness block — surfaces silent broker outages on the report.
        if broker_status:
            status_lines = format_broker_status_lines(broker_status)
            down = [b for b, s in broker_status.items()
                    if s and s.get('queried') and not s.get('responded')]
            color = 'firebrick' if down else 'dimgray'
            ax.text(0.5, 0.12, '\n'.join(status_lines),
                    ha='center', va='top', fontsize=7, color=color,
                    fontfamily='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # --- Summary table page ---
        if len(summary_df) > 0:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            title = ('Top 30 Candidates — ranked by merit per hour'
                     if has_rate else 'Top 30 Candidates by Merit')
            ax.set_title(title, fontsize=14, pad=20)

            table_df = summary_df.head(30).copy()
            table_df['RA_s'] = table_df.apply(
                lambda r: radec_to_sexagesimal(r['ra'], r['dec'])[0], axis=1)
            table_df['Dec_s'] = table_df.apply(
                lambda r: radec_to_sexagesimal(r['ra'], r['dec'])[1], axis=1)

            # Prefer the TNS name where one exists (matches the LLAMAS plan
            # and the briefing); fall back to a short diaObjectId.
            def _display_name(r):
                tns = r.get('tns_name')
                if isinstance(tns, str) and tns.strip():
                    return tns
                return str(r['diaObjectId'])[-10:]
            table_df['name'] = table_df.apply(_display_name, axis=1)

            display_cols = ['name', 'ddf_field', 'RA_s', 'Dec_s',
                           'peak_mag', 'peak_band', 'delta_t', 'merit',
                           'merit_rate', 'exposure_minutes',
                           'brokers_detected', 'fit_method', 'surveys']
            display_df = table_df[
                [c for c in display_cols if c in table_df.columns]
            ].copy()
            # Drop columns that are entirely empty (e.g. ddf_field in wide mode)
            display_df = display_df.dropna(axis=1, how='all')

            # Format numbers
            for col in ['peak_mag', 'merit', 'merit_rate', 'sn_score']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f'{x:.2f}' if pd.notna(x) and np.isfinite(x) else '--')
            if 'delta_t' in display_df.columns:
                display_df['delta_t'] = display_df['delta_t'].apply(
                    lambda x: f'{x:+.1f}d' if pd.notna(x) and np.isfinite(x) else '--')
            if 'exposure_minutes' in display_df.columns:
                display_df['exposure_minutes'] = display_df['exposure_minutes'].apply(
                    lambda x: f'{x:.0f}m' if pd.notna(x) and np.isfinite(x) else '--')
            display_df = display_df.rename(columns={
                'name': 'Object', 'merit_rate': 'merit/hr',
                'exposure_minutes': 'exp est.'})

            tbl = ax.table(
                cellText=display_df.values,
                colLabels=display_df.columns,
                loc='center',
                cellLoc='center',
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(6.5)
            tbl.auto_set_column_width(range(len(display_df.columns)))
            tbl.scale(1.0, 1.12)

            if has_rate:
                ax.text(0.5, -0.06,
                        'merit/hr = merit x (45 min / exp)^0.5. "exp est." is the '
                        'ranking\'s magnitude-based estimate; the LLAMAS plan '
                        're-sizes exposures with redshift where known (see the '
                        'Observing Plan page).',
                        ha='center', va='top', fontsize=8, color='dimgray',
                        transform=ax.transAxes)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # --- Observing Plan page (the executable LLAMAS timeline) ---
        if orch_dir and os.path.isdir(orch_dir):
            date_stamp = obs_date.replace('-', '')
            timeline_path = os.path.join(
                orch_dir, f'LLAMAS_{date_stamp}_timeline.txt')
            acct_path = os.path.join(orch_dir, 'time_accounting.json')
            if os.path.exists(timeline_path):
                with open(timeline_path) as f:
                    timeline_text = f.read()
                charge_lines = []
                try:
                    with open(acct_path) as f:
                        acct = json.load(f)
                    per_prog = {}
                    for entry in acct.get('charge_log', []):
                        if entry.get('date') == obs_date:
                            per_prog[entry['program']] = (
                                per_prog.get(entry['program'], 0.0)
                                + entry.get('hours', 0.0))
                    charge_lines = [f'{prog}: {hrs:.1f} h charged'
                                    for prog, hrs in sorted(per_prog.items())]
                except Exception:
                    pass

                # Map timeline IDs to TNS names so the plan reads the same
                # as the candidate table.
                alias_lines = []
                if 'tns_name' in summary_df.columns:
                    for _, r in summary_df.iterrows():
                        tns = r.get('tns_name')
                        did = str(r['diaObjectId'])
                        if (isinstance(tns, str) and tns.strip()
                                and did in timeline_text):
                            typ = r.get('tns_type')
                            tag = (f' ({typ})' if isinstance(typ, str)
                                   and typ.strip() else '')
                            alias_lines.append(f'{did} = {tns}{tag}')
                if alias_lines:
                    timeline_text += ('\n\nTarget names:\n  '
                                      + '\n  '.join(alias_lines))

                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                ax.set_title(f'LLAMAS Observing Plan — {obs_date}',
                             fontsize=16, fontweight='bold', pad=20)
                ax.text(0.02, 0.96, timeline_text, ha='left', va='top',
                        fontsize=7.5, fontfamily='monospace',
                        transform=ax.transAxes)
                footer = ('Scheduled by the LLAMAS orchestrator (the single '
                          'scheduling authority): composite priority, airmass, '
                          'standards, per-program time accounting.')
                if charge_lines:
                    footer += '\n' + '  |  '.join(charge_lines)
                ax.text(0.02, 0.04, footer, ha='left', va='bottom',
                        fontsize=8, color='dimgray', fontfamily='monospace',
                        transform=ax.transAxes)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # --- Merit Breakdown table pages (rank order, paginated) ---
        has_breakdown = all(c in summary_df.columns for c in ['w_time', 'w_mag', 'w_prob'])
        has_salt_weight = 'w_salt' in summary_df.columns
        if len(summary_df) > 0 and has_breakdown:
            # Same order as the ranking table — do NOT re-sort by raw merit,
            # which would contradict the merit-per-hour rank.
            table_df = summary_df.head(30).copy()

            breakdown_cols = ['diaObjectId', 'merit', 'w_time', 'w_mag', 'w_prob',
                             'w_iaspec', 'w_host', 'w_ext', 'w_broker', 'w_moon']
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
                'w_iaspec': 'W_iasp',
                'w_host': 'W_host',
                'w_ext': 'W_ext',
                'w_broker': 'W_brok',
                'w_moon': 'W_moon',
                'w_salt': 'W_salt',
                'w_absmag': 'W_abs',
            }
            breakdown_df.columns = [col_names.get(c, c) for c in breakdown_df.columns]
            breakdown_df.insert(0, 'Rank', range(1, len(breakdown_df) + 1))

            formula = ('Merit = ' + ' × '.join(
                c for c in breakdown_df.columns if c.startswith('W_')))
            legend_text = (
                'W_time: Gaussian in phase (τ=10d)  |  W_mag: soft top-hat over the LLAMAS window  |  '
                'W_prob: P(SN) classifier\n'
                'W_iasp: Ia-specific evidence [0.8-1.2]  |  W_host: Elliptical=1.0, Spiral=0.6  |  '
                'W_ext: Galactic extinction  |  W_brok: multi-broker bonus\n'
                'W_moon: moon separation penalty  |  W_salt: SALT2 chi2/dof quality [0.5-1.2]  |  '
                'W_abs: absolute mag ~ -19.3\n'
                'Rows in ranking order (merit per hour); Merit alone is the raw science weight.'
            )

            rows_per_page = 15
            for pstart in range(0, len(breakdown_df), rows_per_page):
                chunk = breakdown_df.iloc[pstart:pstart + rows_per_page]
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                ax.set_title(
                    f'Merit Breakdown — ranks {pstart + 1}-'
                    f'{pstart + len(chunk)}', fontsize=14, pad=20)
                ax.text(0.5, 0.92, formula,
                       ha='center', va='top', fontsize=9, fontfamily='monospace',
                       transform=ax.transAxes)
                tbl = ax.table(
                    cellText=chunk.values,
                    colLabels=chunk.columns,
                    loc='center',
                    cellLoc='center',
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(7.5)
                tbl.auto_set_column_width(range(len(chunk.columns)))
                tbl.scale(1.0, 1.4)
                ax.text(0.5, 0.02, legend_text, ha='center', va='bottom',
                       fontsize=7.5, color='dimgray', transform=ax.transAxes)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # --- Merit Function Reference page ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Merit Function Reference', fontsize=16, fontweight='bold', pad=20)

        merit_text = """
MERIT FUNCTION

    Merit = W_time × W_mag × W_prob × W_iaspec × W_host × W_ext × W_broker
            × W_moon × W_salt × W_absmag

Each component ranges from 0 to ~1 (a few reach 1.2). The multiplicative
structure means a candidate needs to score well on ALL factors to rank highly.

RANKING = MERIT PER HOUR

    merit_rate = Merit × (45 min / exposure)^0.5

Targets are ordered by merit_rate: science value weighed against the time it
costs, so a fast bright target outranks a marginally better slow one. The
exponent (--rank-alpha, default 0.5) is a policy knob; 0 recovers pure merit.
Merit itself stays a pure science weight. Exposure here is the magnitude-based
estimate; the LLAMAS plan re-sizes with redshift where known.

COMPONENT DEFINITIONS

W_time  — Phase: exp(−(Δt−Δt_pref)² / 2τ²), τ = 10 d (Δt_pref per program)
    Supernovae are most valuable for spectroscopy near peak brightness.

W_mag   — Magnitude Suitability: soft top-hat — full weight for
    18.0 ≤ m ≤ 21.0 AB, gentle Gaussian roll-offs outside (bright σ = 2.0,
    faint σ = 1.0 mag). Brightness is never punished sharply; too-faint is.

W_prob  — SN Probability: P(SN) from ML classifier [0.1, 1.0]
    From ALeRCE or Fink. ANTARES-only use proxy capped at 0.50.

W_iaspec — Ia-specific Evidence [0.8, 1.2]: TNS spec-type / ALeRCE SNIa /
    early-Ia scores. Positive Ia evidence scores at or above neutral.

W_host  — Host Galaxy Morphology: Elliptical=1.0, Spiral=0.6, Unknown=0.7
    SNe Ia in elliptical hosts have lower Hubble diagram scatter.

W_ext   — Galactic Extinction Penalty: exp(−E(B−V) / 0.15)
    Heavily penalizes targets behind significant Milky Way dust.

W_broker — Multi-broker Agreement, coverage-aware: bonus scales with the
    fraction of ACHIEVABLE brokers that detected the object [1.0, 1.3].

W_moon  — Moon Proximity Penalty [0.3, 1.0]: separation- and
    illumination-dependent; neutral when the moon is down or dark.

W_salt  — SALT2 Template Fit Quality: sigmoid(chi2/dof) [0.5, 1.2]
    Good SALT2 fit (chi2/dof < 2) indicates SN Ia template match.
    Bonus for excellent fits, penalty for poor fits.

W_absmag — Absolute Magnitude: Gaussian at M_B = −19.3, σ = 0.7
    SNe Ia have M_B ~ −19.3 ± 0.5. Requires host redshift.
    Penalizes candidates with absolute mag inconsistent with SN Ia.
"""
        ax.text(0.05, 0.99, merit_text, ha='left', va='top',
                fontsize=8, fontfamily='monospace', transform=ax.transAxes)

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
        # Keep the ranking order (merit per hour when available) so plot
        # rank numbers match the summary table and the LLAMAS plan inputs.
        ordered = (summary_df if has_rate else
                   summary_df.sort_values('merit', ascending=False,
                                          na_position='last'))
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
                    if has_rate and pd.notna(r.get('merit_rate')):
                        info += f"  Merit/hr={r['merit_rate']:.3f}"
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


def generate_observing_schedule(plan_df, mjd_now, obs_date, output_path,
                                broker_status=None, viability_floor=0.01):
    """DEPRECATED (2026-07): no longer called by the nightly pipeline.

    The pipeline's own greedy schedule was retired in favor of the LLAMAS
    orchestrator, the single scheduling authority (see main() Step 9). Kept
    temporarily for reference; remove after one clean release cycle.

    Generate a human-readable observing schedule ordered by priority.

    Uses exposure time estimates if available, otherwise assumes 30 min.
    Lists targets in priority order with coordinates, magnitude, merit,
    optimal observing time, and estimated exposure. Includes merit breakdown
    and a per-broker liveness block.

    The schedule is capped to the night: targets are listed as TONIGHT only
    while cumulative exposure fits within the dark hours; the remainder — and
    any target below ``viability_floor`` in merit — is listed under BACKUP.
    If nothing clears the floor, the schedule says so explicitly instead of
    dressing up unviable targets as a plan.
    """
    if len(plan_df) == 0:
        return

    # Use the priority ordering from plan_df (don't re-sort)
    df = plan_df.reset_index(drop=True)

    # Dark hours available tonight (nautical twilight to twilight)
    dark_hours = None
    try:
        from core.magellan_planning import _get_twilight_times
        evening, morning = _get_twilight_times(obs_date)
        if evening is not None and morning is not None:
            dark_hours = float((morning - evening).to_value('hr'))
    except Exception as e:
        logger.warning("Twilight calculation failed (%s); schedule not "
                       "night-capped", e)

    # Calculate total observing time from exposure estimates or 30 min default
    has_exp = 'exposure_minutes' in df.columns
    if has_exp:
        exp_vals = df['exposure_minutes'].fillna(30).values
    else:
        exp_vals = np.full(len(df), 30.0)
    total_hours = exp_vals.sum() / 60

    # Partition into TONIGHT (viable + fits in the night) and BACKUP
    merit_vals = pd.to_numeric(df.get('merit', pd.Series(np.nan, index=df.index)),
                               errors='coerce').values
    if 'merit_rank' not in df.columns:
        df['merit_rank'] = pd.Series(merit_vals, index=df.index).rank(
            ascending=False, method='min', na_option='bottom')
    viable = np.isfinite(merit_vals) & (merit_vals >= viability_floor)
    tonight_mask = np.zeros(len(df), dtype=bool)
    cum_hours = 0.0
    cap_hours = dark_hours if dark_hours is not None else np.inf
    for i in range(len(df)):
        if not viable[i]:
            continue
        if cum_hours + exp_vals[i] / 60 <= cap_hours:
            tonight_mask[i] = True
            cum_hours += exp_vals[i] / 60
    n_tonight = int(tonight_mask.sum())

    # Check for merit breakdown columns
    has_breakdown = all(c in df.columns for c in ['w_time', 'w_mag', 'w_prob'])

    lines = []
    lines.append(f'# Magellan Observing Schedule — {obs_date} UT')
    dark_s = f'{dark_hours:.1f}h dark' if dark_hours is not None else 'dark hours n/a'
    lines.append(f'# MJD {mjd_now:.1f} | {len(df)} candidates | {dark_s} | '
                 f'{n_tonight} scheduled tonight (~{cum_hours:.1f}h) | '
                 f'{len(df) - n_tonight} backup/overflow')
    lines.append(f'# Sorted by priority: time-critical > setting soon > merit')
    lines.append('#')

    # Broker-liveness block — distinguishes a down broker from an empty sky.
    if broker_status is not None:
        for line in format_broker_status_lines(broker_status, prefix='# '):
            lines.append(line)
        lines.append('#')

    if n_tonight == 0:
        lines.append('# *** NO VIABLE TARGETS TONIGHT ***')
        lines.append(f'# No candidate clears the merit floor ({viability_floor}).')
        lines.append('# Consider standards, manual/ToO targets, or standing down.')
        lines.append('# All candidates are listed below as BACKUP for reference.')
        lines.append('#')

    header = (f'# {"#":>3s}  {"Object":20s}  {"RA":>11s}  {"Dec":>10s}  '
              f'{"PkMag":>6s}  {"dt":>6s}  {"Merit":>6s}  {"Rank":>4s}  '
              f'{"OptUT":>5s}  {"Exp":>5s}  {"DDF":>8s}')
    rule = (f'# {"---":>3s}  {"----":20s}  {"--":>11s}  {"---":>10s}  '
            f'{"-----":>6s}  {"--":>6s}  {"-----":>6s}  {"----":>4s}  '
            f'{"-----":>5s}  {"---":>5s}  {"---":>8s}')

    def _row_line(i, row):
        ra_s, dec_s = radec_to_sexagesimal(row['ra'], row['dec'])
        pmag = f"{row['peak_mag']:.1f}" if np.isfinite(row.get('peak_mag', np.nan)) else '--'
        dt = f"{row['delta_t']:+.0f}d" if np.isfinite(row.get('delta_t', np.nan)) else '--'
        merit = f"{row['merit']:.3f}" if np.isfinite(row.get('merit', np.nan)) else '--'
        rank = (f"{int(row['merit_rank'])}" if np.isfinite(row.get('merit_rank', np.nan))
                else '--')
        ddf = str(row.get('ddf_field', '') or '')
        did = str(row['diaObjectId'])[-12:]
        opt_ut = str(row.get('optimal_time_ut', '--') or '--')
        exp_min = row.get('exposure_minutes', np.nan)
        exp_str = f"{exp_min:.0f}m" if np.isfinite(exp_min) else '30m'
        return (f'  {i:3d}  {did:20s}  {ra_s:>11s}  {dec_s:>10s}  '
                f'{pmag:>6s}  {dt:>6s}  {merit:>6s}  {rank:>4s}  '
                f'{opt_ut:>5s}  {exp_str:>5s}  {ddf:>8s}')

    if n_tonight > 0:
        lines.append('# === TONIGHT ===')
        lines.append(header)
        lines.append(rule)
        k = 0
        for i, (_, row) in enumerate(df.iterrows()):
            if tonight_mask[i]:
                k += 1
                lines.append(_row_line(k, row))

    if (~tonight_mask).any():
        lines.append('#')
        reason = ('below merit floor or beyond tonight\'s dark hours'
                  if n_tonight > 0 else 'all below merit floor')
        lines.append(f'# === BACKUP / OVERFLOW ({reason}) ===')
        lines.append(header)
        lines.append(rule)
        k = 0
        for i, (_, row) in enumerate(df.iterrows()):
            if not tonight_mask[i]:
                k += 1
                lines.append(_row_line(k, row))

    lines.append('#')
    lines.append(f'# Scheduled tonight: ~{cum_hours:.1f}h of {dark_s}; '
                 f'full candidate list would need ~{total_hours:.1f}h')

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
        lines.append('#   W_mag   : soft top-hat       Full weight 18<=m<=21, gentle roll-offs')
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
            # Read w_moon from the SAME breakdown that produced merit (single
            # source of truth), not from moon_penalty directly.
            w_moon = f"{row['w_moon']:.3f}" if np.isfinite(row.get('w_moon', np.nan)) else '1.000'

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
                        help='Enable SALT2 template fitting (requires sncosmo). '
                             'Default: ON in --sky-mode wide, opt-in in ddf mode')
    parser.add_argument('--no-salt', action='store_true',
                        help='Disable SALT2 template fitting (overrides the '
                             'wide-mode default and --use-salt)')
    parser.add_argument('--salt-rescue-cap', type=int, default=30,
                        help='Max SALT rescue fits per run for objects that '
                             'failed the Villar/parabola gate (default: 30)')
    parser.add_argument('--no-redshift', action='store_true',
                        help='Skip NED redshift queries')

    # Sky mode: 'ddf' = legacy latest-N alert sampling (DDF-cone secondary
    # brokers); 'wide' = payload-level target-space selection across the whole
    # Magellan-visible sky (proposal: r<=21.5, z<=0.4, dec<=+22)
    parser.add_argument('--sky-mode', choices=['ddf', 'wide'], default='ddf',
                        help='Candidate selection mode (default: ddf)')
    parser.add_argument('--max-mag', type=float, default=WIDE_MAX_MAG,
                        help='Wide mode: faint limit on brightest detection '
                             f'(default: {WIDE_MAX_MAG})')
    parser.add_argument('--max-z', type=float, default=WIDE_MAX_Z,
                        help=f'Wide mode: max redshift (default: {WIDE_MAX_Z})')
    parser.add_argument('--dec-limit', type=float, default=WIDE_DEC_LIMIT,
                        help='Wide mode: max declination in deg '
                             f'(default: +{WIDE_DEC_LIMIT}, airmass 1.6 from LCO)')
    parser.add_argument('--fit-cap', type=int, default=WIDE_FIT_CAP,
                        help='Wide mode: max objects sent to photometry+fitting '
                             f'(default: {WIDE_FIT_CAP})')
    parser.add_argument('--no-z-enrich', action='store_true',
                        help='Skip the post-ranking TNS+NED redshift '
                             'enrichment of finalists (wide mode)')
    parser.add_argument('--no-tournament', action='store_true',
                        help='Skip the multi-type template tournament '
                             '(SALT2 vs Ibc/IIP/IIn typing evidence on '
                             'finalists; wide mode, needs SALT enabled)')

    # Scheduling: the LLAMAS orchestrator is the single scheduling authority
    parser.add_argument('--allocations', default=DEFAULT_ALLOCATIONS,
                        help='allocations YAML for the LLAMAS orchestrator '
                             f'(default: {DEFAULT_ALLOCATIONS} — EXAMPLE budgets)')
    parser.add_argument('--moon-phase', choices=['dark', 'grey', 'bright'],
                        default=None,
                        help='Override the moon phase passed to the '
                             'orchestrator (default: derived from tonight\'s '
                             'moon illumination)')
    parser.add_argument('--no-orchestrate', action='store_true',
                        help='Stop after candidates.csv; do not generate the '
                             'LLAMAS observing plan')
    parser.add_argument('--rank-alpha', type=float, default=0.5,
                        help='Merit-per-hour exponent: ranking orders by '
                             'merit x (45min/exposure)^alpha — value density. '
                             '0 ranks by pure merit (default: 0.5)')
    parser.add_argument('--max-phase', type=float, default=25.0,
                        help='Max |days from fitted peak| to keep a candidate '
                             '(default: 25, matched to the tau=10d merit scale)')
    parser.add_argument('--max-baseline', type=float, default=150.0,
                        help='Reject objects whose high-SNR detections span more '
                             'days than this (long-lived variable/AGN; default: 150)')

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
    if args.sky_mode == 'wide':
        logger.info("Sky mode: WIDE (r<=%.1f, z<=%.1f, dec<=%+.0f, fit cap %d)",
                    args.max_mag, args.max_z, args.dec_limit, args.fit_cap)
        logger.info("Wide brokers: Fink LSST + live ALeRCE-ZTF "
                    "(ANTARES/ALeRCE-LSST remain DDF-cone only)")
    candidates, broker_status = fetch_all_broker_candidates(
        fink if fink_available else None,
        min_prob=args.min_prob,
        days_back=args.days_back,
        # Wide mode needs a deep slice of the whole-sky stream to select from;
        # the legacy default only skims the latest few hundred alerts.
        n_fetch=max(args.max_candidates, 2000) if args.sky_mode == 'wide'
                else args.max_candidates,
        sky_mode=args.sky_mode,
        mjd_now=mjd_now,
        wide_kwargs={'dec_limit': args.dec_limit, 'max_mag': args.max_mag,
                     'max_z': args.max_z, 'fit_cap': args.fit_cap},
        fink_only=args.fink_only,
    )

    # Emit broker-liveness sidecar early so it survives even an early exit.
    try:
        write_broker_status(broker_status, night_dir)
    except Exception as e:
        logger.warning("Could not write broker_status.json: %s", e)
    log_broker_status(broker_status)

    if len(candidates) == 0:
        logger.error("No candidates found")
        sys.exit(1)

    # --- Step 2a: TNS cross-match (identify already-reported transients) ---
    do_tns = not args.no_tns if hasattr(args, 'no_tns') else True
    if do_tns and args.sky_mode == 'wide':
        # Fink already cross-matches TNS server-side (f:xm_tns_*). Reusing the
        # payload avoids the standalone client's one-by-one rate-limited
        # queries (~100/min, 60s penalty sleeps) for identical information.
        def _xm(col):
            return candidates.get(col, pd.Series('', index=candidates.index))
        candidates['tns_name'] = _xm('tns_name_xm')
        candidates['tns_type'] = _xm('tns_type_xm').replace('', np.nan)
        candidates['tns_redshift'] = pd.to_numeric(
            candidates.get('z_tns', pd.Series(dtype=float, index=candidates.index)),
            errors='coerce')
        candidates['tns_match'] = (
            _xm('tns_name_xm').fillna('').astype(str).str.strip() != '')
        logger.info("TNS via Fink payload: %d/%d already reported, "
                    "%d spectroscopically classified",
                    int(candidates['tns_match'].sum()), len(candidates),
                    int(candidates['tns_type'].notna().sum()))
        do_tns = False
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

    # SALT2 fitting: ON by default in wide mode, opt-in (--use-salt) in ddf
    # mode, --no-salt always wins. NED redshifts feed SALT, so resolve first.
    want_salt = (args.use_salt or args.sky_mode == 'wide') and not args.no_salt
    do_salt = resolve_salt_mode(args.sky_mode, args.use_salt, args.no_salt)
    do_redshift = not args.no_redshift and HAS_NED
    if want_salt and not HAS_SNCOSMO:
        logger.warning("SALT2 fitting requested but sncosmo not installed — skipping SALT fits")
    if do_salt:
        # Preflight: constructing the model may trigger a one-time template
        # download. Do it exactly once here; per-object fits reuse the cache.
        if load_salt_model() is None:
            logger.warning("*" * 70)
            logger.warning("SALT2 model 'salt2-extended' UNAVAILABLE — proceeding "
                           "with SALT fitting DISABLED. If this machine is "
                           "offline, pre-warm the cache with "
                           "`python -c \"import sncosmo; "
                           "sncosmo.Model(source='salt2-extended')\"` or point "
                           "SNCOSMO_DATA_DIR at a warmed data directory.")
            logger.warning("*" * 70)
            do_salt = False
    if want_salt:
        logger.info("SALT2 fitting: %s (rescue cap %d)",
                    "enabled" if do_salt else "disabled",
                    args.salt_rescue_cap)

    # --- Step 3a: Query NED redshifts (needed for SALT fitting and absolute mag) ---
    redshifts = {}  # did -> {redshift, distmod, ned_name, separation_arcsec}
    if args.sky_mode == 'wide' and 'z_best' in candidates.columns:
        # Wide mode: redshifts come from the Fink payload (TNS spec-z when
        # available, else Legacy DR8 host photo-z) — no NED nearest-neighbor
        # guessing, no per-object NED queries.
        from astropy.cosmology import FlatLambdaCDM
        _cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
        for _, row in candidates.iterrows():
            z = row.get('z_best')
            if pd.notna(z) and 0.001 < z < 3.0:
                redshifts[row['diaObjectId']] = {
                    'redshift': float(z),
                    'distmod': float(_cosmo.distmod(float(z)).value),
                    'ned_name': f"payload:{row.get('z_source', 'unknown')}",
                    'separation_arcsec': np.nan,
                }
        logger.info("Payload redshifts: %d/%d candidates have z "
                    "(TNS spec-z or Legacy photo-z)",
                    len(redshifts), len(candidates))
        do_redshift = False
    if do_redshift:
        logger.info("Querying NED for host galaxy redshifts (with caching)...")
        # Use batch function which handles caching
        from cache.alert_cache import AlertCache
        ned_cache = AlertCache()
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

    # Build redshift info for SALT fitting: z + provenance. The provenance
    # (z_source) decides fixed-z vs bounded free-z via salt_z_policy().
    zsrc_lookup = {}
    if 'z_source' in candidates.columns:
        zsrc_lookup = {d: str(s or '') for d, s in
                       zip(candidates['diaObjectId'],
                           candidates['z_source'].fillna(''))}
    z_info_for_salt = {}
    for did, info in redshifts.items():
        zv = info.get('redshift')
        if zv is None or not np.isfinite(zv) or zv <= 0:
            continue
        src = zsrc_lookup.get(did, '')
        if not src:
            # DDF mode: z came from a NED host association (spec-z) — fix it.
            src = 'tns_specz'
        z_info_for_salt[did] = {'z': float(zv), 'source': src}

    # E(B-V) lookup for the SALT MW-dust component
    ebv_lookup = {}
    ebv_col = next((c for c in ('E_BV', 'ebv') if c in candidates.columns), None)
    if ebv_col is not None:
        for did, ebv in zip(candidates['diaObjectId'],
                            pd.to_numeric(candidates[ebv_col], errors='coerce')):
            if np.isfinite(ebv):
                ebv_lookup[did] = float(ebv)

    fit_results = fetch_and_fit(fink if fink_available else None,
                                candidates, mjd_now,
                                fetch_ztf=do_ztf and HAS_ALERCE,
                                fetch_atlas=do_atlas and HAS_ATLAS,
                                min_snr_points=args.min_snr_points,
                                min_bands=args.min_bands,
                                min_fit_bands=args.min_fit_bands,
                                prefilter_min_sources=args.prefilter_min_sources,
                                use_salt=do_salt,
                                redshifts=z_info_for_salt,
                                max_rise_time=args.max_rise_time,
                                max_phase_days=args.max_phase,
                                max_baseline_days=args.max_baseline,
                                ebv_lookup=ebv_lookup,
                                salt_rescue_cap=args.salt_rescue_cap)
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

    # --- Step 5a: redshift enrichment for the finalists (wide mode) ---
    # ZTF-stream targets carry no payload z; a post-ranking TNS+NED pass over
    # the ~30 finalists is cheap and fixes exposure estimates + abs-mag.
    if args.sky_mode == 'wide' and not args.no_z_enrich and len(summary) > 0:
        try:
            summary = enrich_finalist_redshifts(summary, fit_results,
                                                use_salt=do_salt)
        except Exception as e:
            logger.warning("Redshift enrichment failed (continuing): %s", e)

    # --- Step 5b: template-tournament typing for the finalists (wide mode) ---
    # SALT2 vs CC templates on each finalist: positive typing evidence
    # (which template wins), not just "poor Ia fit". Runs after z-enrichment
    # so gained spectroscopic redshifts are fixed in the comparison fits.
    if (args.sky_mode == 'wide' and do_salt and not args.no_tournament
            and len(summary) > 0):
        try:
            summary = enrich_finalist_typing(summary, fit_results)
        except Exception as e:
            logger.warning("Template tournament failed (continuing): %s", e)

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
            # Fold the night's moon penalty into the ranking merit (and w_moon)
            # now that moon_penalty has been computed; this re-sorts by merit.
            plan = recompute_merit_with_moon(plan)
        except Exception as e:
            logger.warning("Observability calculation failed: %s. Using all targets.", e)
            plan = summary.copy()
    else:
        plan = summary.copy()

    # Propagate the moon-aware merit/breakdown back onto the summary frame so
    # candidates.csv and the PDF report rank/report identically to the plan.
    moon_cols = ['merit', 'merit_exotic', 'w_time', 'w_mag', 'w_prob',
                 'w_host', 'w_ext', 'w_broker', 'w_moon', 'w_salt',
                 'w_absmag', 'w_iaspec',
                 'moon_penalty', 'moon_separation', 'moon_illumination',
                 # needed by the merit-per-hour ranking (merit_rate)
                 'exposure_minutes', 'optimal_time_ut']
    avail_cols = [c for c in moon_cols if c in plan.columns]
    if avail_cols and 'diaObjectId' in plan.columns:
        moon_lookup = plan.set_index('diaObjectId')[avail_cols]
        for col in avail_cols:
            summary[col] = summary['diaObjectId'].map(moon_lookup[col])
        summary = summary.sort_values('merit', ascending=False, na_position='last')
        summary = summary.reset_index(drop=True)

    # Within-night ranking. `merit` stays the PURE science value (every
    # factor auditable); the RANKING orders by value DENSITY:
    #     merit_rate = merit x (45 min / exposure)^alpha
    # (PI decision 2026-07-14: a target delivering its science in a fifth of
    # the time should outrank a marginally-better one that eats a quarter of
    # the night; alpha=0 restores pure-merit ordering). merit_rank/percentile
    # are what the observer acts on; 1 = best.
    if 'merit' in summary.columns and len(summary) > 0:
        alpha = getattr(args, 'rank_alpha', 0.5)
        exp_col = pd.to_numeric(summary.get('exposure_minutes'), errors='coerce')
        if alpha > 0 and exp_col is not None and exp_col.notna().any():
            density = (RATE_REF_MINUTES / exp_col.clip(lower=5.0)) ** alpha
            density = density.fillna(1.0)
        else:
            density = 1.0
        summary['merit_rate'] = summary['merit'] * density
        if 'merit_exotic' in summary.columns:
            summary['merit_exotic_rate'] = summary['merit_exotic'] * density
        m = summary['merit_rate']
        summary['merit_rank'] = m.rank(ascending=False, method='min', na_option='bottom').astype(int)
        n_ranked = m.notna().sum()
        if n_ranked > 1:
            summary['merit_percentile'] = ((n_ranked - summary['merit_rank'])
                                           / (n_ranked - 1) * 100).clip(0, 100).round(1)
        else:
            summary['merit_percentile'] = np.where(m.notna(), 100.0, np.nan)
        if 'merit_exotic_rate' in summary.columns:
            summary['merit_exotic_rank'] = summary['merit_exotic_rate'].rank(
                ascending=False, method='min', na_option='bottom').astype(int)
        summary = summary.sort_values('merit_rate', ascending=False,
                                      na_position='last').reset_index(drop=True)

    # --- Step 7: Generate light curve plots ---
    plot_paths = generate_light_curve_plots(fit_results, lc_dir, summary)

    # --- Step 8: Save outputs ---
    # Summary CSV
    csv_path = os.path.join(night_dir, 'candidates.csv')
    summary.to_csv(csv_path, index=False)
    logger.info("Candidates CSV: %s", csv_path)

    # --- Step 9: THE schedule — the LLAMAS orchestrator (single authority) ---
    # The pipeline RANKS; the orchestrator SCHEDULES. The pipeline's own
    # greedy schedule / TCS catalog / slew-optimized sequence outputs were
    # retired 2026-07: they re-implemented exposure estimation, observability
    # and sequencing with rules that diverged from the orchestrator's
    # (hard airmass limit, quartile weights, standards, time accounting).
    orch_dir = None
    if not args.no_orchestrate and len(summary) > 0:
        try:
            from orchestrator.run_nightly import run_nightly as orch_run_nightly
            moon_illum = (summary['moon_illumination'].iloc[0]
                          if 'moon_illumination' in summary.columns else np.nan)
            moon_phase = args.moon_phase or (
                'dark' if (np.isfinite(moon_illum) and moon_illum < 0.25) else
                'grey' if (np.isfinite(moon_illum) and moon_illum < 0.65) else
                'bright' if np.isfinite(moon_illum) else 'grey')
            if args.allocations == DEFAULT_ALLOCATIONS:
                logger.warning("Using EXAMPLE allocations (%s) — budgets are "
                               "illustrative until MAGNETS agrees real numbers",
                               args.allocations)
            orch_dir = os.path.join(night_dir, 'llamas')
            orch_run_nightly(date=obs_date, candidates_path=csv_path,
                             allocations_path=args.allocations,
                             moon_phase=moon_phase, output_dir=orch_dir)
            logger.info("LLAMAS plan (scheduling authority): %s", orch_dir)
        except Exception as e:
            logger.error("Orchestrator scheduling failed: %s "
                         "(candidates.csv is still valid; run "
                         "`python -m orchestrator run-nightly` manually)", e)
            orch_dir = None

    # PDF report (ranking + light curves; the executable plan lives in llamas/)
    pdf_path = os.path.join(night_dir, f'report_{ut_stamp}.pdf')
    generate_pdf_report(summary, fit_results, plot_paths,
                        pdf_path, mjd_now, obs_date,
                        observing_sequence=None,
                        broker_status=broker_status,
                        orch_dir=orch_dir)

    # --- Done ---
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Night directory: %s", night_dir)
    logger.info("  candidates.csv         %d ranked candidates", len(summary))
    if orch_dir:
        logger.info("  llamas/                executable LLAMAS plan "
                    "(timeline, catalog, summary, accounting)")
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

    # Print top 5 by merit (ranking preview; the schedule lives in llamas/)
    if len(summary) > 0:
        top = summary.sort_values('merit', ascending=False).head(5)
        logger.info("\nTop 5 by merit:")
        for _, r in top.iterrows():
            ra_s, dec_s = radec_to_sexagesimal(r['ra'], r['dec'])
            logger.info("  %s  %s %s  mag=%.1f  dt=%+.0fd  merit=%.3f  %s",
                        str(r['diaObjectId'])[-12:], ra_s, dec_s,
                        r['peak_mag'] if np.isfinite(r['peak_mag']) else 0,
                        r['delta_t'] if np.isfinite(r['delta_t']) else 0,
                        r['merit'] if np.isfinite(r['merit']) else 0,
                        r.get('ddf_field', ''))


if __name__ == '__main__':
    main()
