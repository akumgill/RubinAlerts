"""ANTARES alert broker client — DDF-focused transient search."""

import os
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from .base_client import BaseBrokerClient
from core.ddf_fields import DDF_FIELDS, DDF_SEARCH_RADIUS_DEG

logger = logging.getLogger(__name__)

# Default MJD cutoff for recent activity (days before current MJD)
DEFAULT_DAYS_RECENT = 60

# ANTARES ant_survey codes
SURVEY_ZTF_DETECTION = 1
SURVEY_ZTF_UPPER_LIMIT = 2
SURVEY_LSST = 4

SURVEY_NAMES = {
    SURVEY_ZTF_DETECTION: 'ZTF',
    SURVEY_ZTF_UPPER_LIMIT: 'ZTF_upper_limit',
    SURVEY_LSST: 'Rubin',
}

# Quality thresholds for transient candidate selection
MAX_SGSCORE = 0.5       # reject point sources (stars)
MIN_RB_SCORE = 0.5      # reject likely bogus detections
MAX_DURATION_DAYS = 200  # reject persistent variables
MIN_MAG_VALUES = 5      # require enough photometry for fitting
MIN_DURATION_DAYS = 2   # require multi-epoch baseline
MAX_BRIGHTEST_MAG = 23  # reject very faint candidates (hard to type)


class AntaresClient(BaseBrokerClient):
    """Client for querying ANTARES alert broker (antares_client 1.14.0).

    Searches the 6 Rubin Deep Drilling Fields using cone searches,
    then filters for galaxy-associated transients using ztf_sgscore1,
    real/bogus score, and light curve duration.
    """

    def __init__(self, cache_dir: str = './cache/data'):
        super().__init__('ANTARES', cache_dir)
        self._verify_import()
        os.makedirs(cache_dir, exist_ok=True)

    def _verify_import(self):
        try:
            from antares_client import search as _
            self._patch_cone_search_compat()
            logger.info("ANTARES client initialized successfully")
        except ImportError:
            logger.warning("antares-client not available. Install with: pip3 install antares-client")
            raise

    @staticmethod
    def _patch_cone_search_compat():
        """Fix antares_client 1.14 / astropy 7.x incompatibility.

        antares_client calls Quantity.to_string(decimal=True) which was
        removed in astropy 7.0. Monkey-patch the search module to use
        a compatible format string instead.
        """
        import antares_client.search as search_mod
        import inspect
        import astropy.units

        src = inspect.getsource(search_mod.cone_search)
        if 'decimal=True' not in src:
            return  # already compatible

        _original_cone_search = search_mod.cone_search

        def _patched_cone_search(center, radius):
            # Reproduce the original query but avoid decimal= kwarg
            radius_str = f"{radius.to_value(astropy.units.deg)} degree"
            center_str = center.to_string()

            query = {
                "query": {
                    "bool": {
                        "filter": {
                            "sky_distance": {
                                "distance": radius_str,
                                "htm16": {"center": center_str},
                            }
                        }
                    }
                }
            }
            return search_mod.search(query)

        search_mod.cone_search = _patched_cone_search
        logger.info("Patched antares_client cone_search for astropy 7.x compatibility")

    def _get_cache_path(self, query_type: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.cache_dir, f"antares_{query_type}_{timestamp}.json")

    def _load_cache(self, cache_path: str) -> Optional[list]:
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded ANTARES cache from {cache_path}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_cache(self, cache_path: str, data: list):
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, default=str)
                logger.info(f"Saved ANTARES cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _get_locus_cache_path(self) -> str:
        """Get path for persistent locus-level cache."""
        return os.path.join(self.cache_dir, "antares_locus_cache.json")

    def _load_locus_cache(self) -> Dict[str, Dict]:
        """Load persistent locus cache (locus_id -> parsed data)."""
        path = self._get_locus_cache_path()
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} cached ANTARES loci")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load locus cache: {e}")
        return {}

    def _save_locus_cache(self, cache: Dict[str, Dict]):
        """Save persistent locus cache."""
        path = self._get_locus_cache_path()
        try:
            with open(path, 'w') as f:
                json.dump(cache, f, default=str)
        except Exception as e:
            logger.warning(f"Failed to save locus cache: {e}")

    def _search_single_field(self, field: Dict, per_field_limit: int,
                              max_checked: int, require_rubin: bool,
                              mjd_cutoff: Optional[float],
                              locus_cache: Dict[str, Dict],
                              seen_ids: set) -> Tuple[List[Dict], Dict[str, int], int]:
        """Search a single DDF field for transient candidates.

        Returns (alerts_list, rejection_reasons, checked_count).
        """
        from antares_client.search import cone_search
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        field_name = field['name']
        center = SkyCoord(ra=field['ra'] * u.deg, dec=field['dec'] * u.deg)
        radius = DDF_SEARCH_RADIUS_DEG * u.deg

        logger.info(f"  Searching DDF {field_name} "
                   f"(RA={field['ra']:.2f}, Dec={field['dec']:.2f}, "
                   f"r={DDF_SEARCH_RADIUS_DEG:.2f} deg)")

        alerts_list = []
        rejection_reasons = {}
        field_count = 0
        field_accepted = 0
        cache_hits = 0

        try:
            for locus in cone_search(center, radius):
                locus_id = locus.locus_id
                if locus_id in seen_ids:
                    continue
                seen_ids.add(locus_id)
                field_count += 1

                if field_accepted >= per_field_limit:
                    break
                if field_count >= max_checked:
                    logger.info(f"    {field_name}: hit search cap ({max_checked})")
                    break

                if field_count % 500 == 0:
                    logger.info(f"    {field_name}: checked {field_count} loci, "
                               f"accepted {field_accepted} so far...")

                try:
                    # Date pre-filter: skip loci without recent activity
                    if mjd_cutoff is not None:
                        newest_obs = locus.properties.get('newest_alert_observation_time')
                        if newest_obs is not None:
                            try:
                                if float(newest_obs) < mjd_cutoff:
                                    rejection_reasons['old_activity'] = rejection_reasons.get('old_activity', 0) + 1
                                    continue
                            except (TypeError, ValueError):
                                pass

                    # Check locus cache first
                    if locus_id in locus_cache:
                        alert_dict = locus_cache[locus_id].copy()
                        cache_hits += 1
                    else:
                        alert_dict = self._parse_locus(locus, f'ddf_{field_name}')
                        alert_dict['has_rubin'] = self._locus_has_rubin_data(locus)
                        alert_dict['has_ztf'] = self._locus_has_ztf_data(locus)
                        # Cache the parsed locus
                        locus_cache[locus_id] = alert_dict.copy()

                    alert_dict['ddf_field'] = field_name

                    # Quality filters
                    passed, reason = self._passes_quality_cuts(alert_dict, return_reason=True)
                    if not passed:
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                        continue

                    if require_rubin and not alert_dict['has_rubin']:
                        rejection_reasons['no_rubin'] = rejection_reasons.get('no_rubin', 0) + 1
                        continue

                    alerts_list.append(alert_dict)
                    field_accepted += 1

                except Exception as e:
                    logger.debug(f"Failed to parse locus {locus_id}: {e}")

        except Exception as e:
            logger.warning(f"Error searching DDF {field_name}: {e}")

        # Log rejection breakdown
        if field_count > 100 and field_accepted < per_field_limit // 2:
            reasons_str = ', '.join(f"{k}={v}" for k, v in sorted(
                rejection_reasons.items(), key=lambda x: -x[1])[:5])
            logger.info(f"    {field_name}: low acceptance — rejections: {reasons_str}")

        logger.info(f"    {field_name}: checked {field_count}, accepted {field_accepted}"
                   + (f" ({cache_hits} cache hits)" if cache_hits > 0 else ""))

        return alerts_list, rejection_reasons, field_count

    def query_alerts(self,
                    class_name: str = 'SN Ia',
                    min_probability: float = 0.5,
                    days_back: int = 30,
                    limit: int = 200,
                    require_rubin: bool = False,
                    ddf_fields: Optional[List[Dict]] = None,
                    parallel: bool = True,
                    max_workers: int = 3) -> pd.DataFrame:
        """
        Query ANTARES for transient candidates in Rubin Deep Drilling Fields.

        Strategy:
        1. Cone search each DDF for loci (in parallel if enabled)
        2. Pre-filter by date (skip loci without recent activity)
        3. Filter for galaxy-associated objects (sgscore < 0.5)
        4. Filter for transient-like duration (< 200 days)
        5. Filter for real detections (rb >= 0.5)
        6. Optionally require Rubin/LSST photometry

        Parameters
        ----------
        parallel : bool
            If True, search DDF fields in parallel (default True)
        max_workers : int
            Max parallel threads for field searches (default 3)
        """
        fields = ddf_fields or DDF_FIELDS
        field_names = ','.join(f['name'] for f in fields)
        cache_key = f"alerts_v5_ddf_{field_names}_d{days_back}_n{limit}_rubin{int(require_rubin)}"
        cache_path = self._get_cache_path(cache_key)
        cached_data = self._load_cache(cache_path)
        if cached_data:
            return pd.DataFrame(cached_data)

        try:
            from astropy.time import Time

            # Compute MJD cutoff for date pre-filtering
            current_mjd = Time.now().mjd
            mjd_cutoff = current_mjd - DEFAULT_DAYS_RECENT

            # Load persistent locus cache
            locus_cache = self._load_locus_cache()
            initial_cache_size = len(locus_cache)

            # Shared set for deduplication across fields
            seen_ids = set()

            # Allocate quota per field
            per_field_limit = max(limit // len(fields), 20)
            max_checked_per_field = 2000

            alerts_list = []
            total_checked = 0

            if parallel and len(fields) > 1:
                logger.info(f"Searching {len(fields)} DDF fields in parallel (max_workers={max_workers})")

                # Thread-safe seen_ids handled via manager or post-merge dedup
                field_results = {}

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._search_single_field,
                            field, per_field_limit, max_checked_per_field,
                            require_rubin, mjd_cutoff, locus_cache, set()
                        ): field['name']
                        for field in fields
                    }

                    for future in as_completed(futures):
                        field_name = futures[future]
                        try:
                            field_alerts, _, field_checked = future.result()
                            field_results[field_name] = field_alerts
                            total_checked += field_checked
                        except Exception as e:
                            logger.warning(f"Field {field_name} search failed: {e}")

                # Merge and deduplicate results
                merged_ids = set()
                for field_name in [f['name'] for f in fields]:  # preserve field order
                    if field_name in field_results:
                        for alert in field_results[field_name]:
                            oid = alert.get('object_id')
                            if oid and oid not in merged_ids:
                                merged_ids.add(oid)
                                alerts_list.append(alert)

            else:
                # Sequential search (original behavior)
                for field in fields:
                    field_alerts, _, field_checked = self._search_single_field(
                        field, per_field_limit, max_checked_per_field,
                        require_rubin, mjd_cutoff, locus_cache, seen_ids
                    )
                    alerts_list.extend(field_alerts)
                    total_checked += field_checked

            logger.info(f"DDF search: checked {total_checked} loci total, "
                       f"accepted {len(alerts_list)} transient candidates")

            # Save updated locus cache if it grew
            if len(locus_cache) > initial_cache_size:
                self._save_locus_cache(locus_cache)
                logger.info(f"Locus cache updated: {initial_cache_size} -> {len(locus_cache)}")

            if alerts_list:
                self._save_cache(cache_path, alerts_list)

            df = pd.DataFrame(alerts_list)
            logger.info(f"Retrieved {len(df)} ANTARES alerts from DDFs")
            return df

        except Exception as e:
            logger.error(f"Error querying ANTARES: {e}")
            return pd.DataFrame()

    def _passes_quality_cuts(self, alert_dict: Dict, return_reason: bool = False):
        """Check if a locus passes quality cuts for transient candidacy.

        Early filtering: reject candidates that won't have enough data
        for light curve fitting, saving downstream processing.

        Parameters
        ----------
        alert_dict : dict
            Parsed locus data.
        return_reason : bool
            If True, return (passed, reason) tuple instead of just bool.

        Returns
        -------
        bool or (bool, str) depending on return_reason.
        """
        # Photometry count: require enough points for fitting
        num_mags = alert_dict.get('num_mag_values', 0)
        if num_mags < MIN_MAG_VALUES:
            return (False, 'few_mags') if return_reason else False

        # Duration: require multi-epoch baseline (not single-night)
        duration = alert_dict.get('duration_days')
        if duration is not None and duration < MIN_DURATION_DAYS:
            return (False, 'short_duration') if return_reason else False

        # Duration: reject persistent variables (too long)
        if duration is not None and duration > MAX_DURATION_DAYS:
            return (False, 'long_duration') if return_reason else False

        # Brightness: reject very faint candidates
        brightest = alert_dict.get('brightest_mag')
        if brightest is not None and brightest > MAX_BRIGHTEST_MAG:
            return (False, 'too_faint') if return_reason else False

        # Star/galaxy separation: reject point sources
        sgscore = alert_dict.get('ztf_sgscore1')
        if sgscore is not None and sgscore >= MAX_SGSCORE:
            return (False, 'star') if return_reason else False

        # Real/bogus: reject likely artifacts
        rb = alert_dict.get('ztf_rb')
        if rb is not None and rb < MIN_RB_SCORE:
            return (False, 'low_rb') if return_reason else False

        return (True, None) if return_reason else True

    def _parse_locus(self, locus, source_tag: str) -> Dict[str, Any]:
        """Parse an ANTARES Locus object into a flat dictionary."""
        props = locus.properties
        coord = locus.coordinates

        survey_info = props.get('survey', {})
        surveys_present = list(survey_info.keys())

        # Compute duration from locus properties
        oldest = props.get('oldest_alert_observation_time')
        newest = props.get('newest_alert_observation_time')
        duration_days = None
        if oldest is not None and newest is not None:
            try:
                duration_days = float(newest) - float(oldest)
            except (TypeError, ValueError):
                pass

        # Extract quality metrics from the most recent alert
        sgscore = None
        distpsnr = None
        rb_score = None
        drb_score = None
        if hasattr(locus, 'alerts') and locus.alerts:
            last_alert = locus.alerts[-1]
            alert_props = last_alert.properties
            sgscore = alert_props.get('ztf_sgscore1')
            distpsnr = alert_props.get('ztf_distpsnr1')
            rb_score = alert_props.get('ztf_rb')
            drb_score = alert_props.get('ztf_drb')

        return {
            'object_id': locus.locus_id,
            'ra': coord.ra.deg,
            'dec': coord.dec.deg,
            'discovery_date': oldest,
            'newest_obs_time': newest,
            'duration_days': duration_days,
            'brightest_mag': props.get('brightest_alert_magnitude'),
            'newest_mag': props.get('newest_alert_magnitude'),
            'num_alerts': props.get('num_alerts', 0),
            'num_mag_values': props.get('num_mag_values', 0),
            'ztf_object_id': props.get('ztf_object_id'),
            'rubin_dia_object_id': self._get_rubin_dia_object_id(locus),
            'tags': ','.join(locus.tags) if locus.tags else '',
            'source_tag': source_tag,
            'anomaly_score': props.get('anomaly_score'),
            'surveys': ','.join(surveys_present),
            'ztf_sgscore1': sgscore,
            'ztf_distpsnr1': distpsnr,
            'ztf_rb': rb_score,
            'ztf_drb': drb_score,
            'broker': 'ANTARES',
        }

    def _locus_has_rubin_data(self, locus) -> bool:
        """Check if a locus has actual Rubin/LSST photometric detections."""
        try:
            lc = locus.lightcurve
            if not isinstance(lc, pd.DataFrame) or lc.empty:
                return False
            if 'ant_survey' not in lc.columns:
                return False
            rubin_rows = lc[lc['ant_survey'] == SURVEY_LSST]
            if rubin_rows.empty:
                return False
            for mag_col in ['ant_mag', 'ant_mag_corrected']:
                if mag_col in rubin_rows.columns and rubin_rows[mag_col].notna().any():
                    return True
            return False
        except Exception:
            return False

    def _get_rubin_dia_object_id(self, locus) -> str:
        """Extract the Rubin/LSST DIA object ID from a locus."""
        try:
            survey_info = locus.properties.get('survey', {})
            if 'lsst' in survey_info:
                lsst_info = survey_info['lsst']
                dia_ids = lsst_info.get('dia_object_id', [])
                if dia_ids:
                    valid = [str(d) for d in dia_ids if d]
                    if valid:
                        return valid[0]
            return ''
        except Exception:
            return ''

    def _locus_has_ztf_data(self, locus) -> bool:
        """Check if a locus has ZTF data points."""
        try:
            survey_info = locus.properties.get('survey', {})
            if 'ztf' in survey_info:
                ztf_info = survey_info['ztf']
                ztf_ids = ztf_info.get('id', [])
                if ztf_ids and any(z for z in ztf_ids if z):
                    return True
            return False
        except Exception:
            return False

    def get_light_curve(self, object_id: str) -> Optional[pd.DataFrame]:
        """Retrieve normalized light curve from ANTARES."""
        try:
            from antares_client.search import get_by_id

            logger.info(f"Retrieving light curve for ANTARES object {object_id}")
            locus = get_by_id(object_id)

            if locus is None:
                logger.warning(f"Locus {object_id} not found")
                return None

            lc = locus.lightcurve
            if not isinstance(lc, pd.DataFrame) or lc.empty:
                logger.warning(f"No lightcurve data for {object_id}")
                return None

            logger.info(f"Raw lightcurve: {len(lc)} rows, columns: {list(lc.columns)}")

            if 'ant_survey' in lc.columns:
                for sv, cnt in lc['ant_survey'].value_counts().items():
                    sv_name = SURVEY_NAMES.get(sv, f'code={sv}')
                    logger.info(f"  {sv_name}: {cnt} rows")

            # Filter to actual detections (exclude upper limits)
            if 'ant_survey' in lc.columns:
                detections = lc[lc['ant_survey'] != SURVEY_ZTF_UPPER_LIMIT].copy()
            else:
                detections = lc.copy()

            if detections.empty:
                return None

            # Build magnitude: prefer ant_mag, fall back to ant_mag_corrected
            mag_values = detections.get('ant_mag', pd.Series(dtype=float)).copy()
            magerr_values = detections.get('ant_magerr', pd.Series(dtype=float)).copy()

            if 'ant_mag_corrected' in detections.columns:
                mask = mag_values.isna()
                if mask.any():
                    mag_values.loc[mask] = detections.loc[mask, 'ant_mag_corrected']
            if 'ant_magerr_corrected' in detections.columns:
                mask = magerr_values.isna()
                if mask.any():
                    magerr_values.loc[mask] = detections.loc[mask, 'ant_magerr_corrected']

            data = {
                'mjd': detections['ant_mjd'].values if 'ant_mjd' in detections.columns else np.nan,
                'magnitude': mag_values.values if len(mag_values) > 0 else np.nan,
                'mag_err': magerr_values.values if len(magerr_values) > 0 else 0.0,
                'band': detections['ant_passband'].values if 'ant_passband' in detections.columns else 'unknown',
            }

            if 'ant_survey' in detections.columns:
                data['survey_code'] = detections['ant_survey'].values
                data['survey'] = detections['ant_survey'].map(SURVEY_NAMES).values
            else:
                data['survey_code'] = 0
                data['survey'] = 'unknown'

            df = pd.DataFrame(data)
            df = df.dropna(subset=['mjd', 'magnitude'])
            df = df.sort_values('mjd').reset_index(drop=True)

            logger.info(f"Retrieved {len(df)} photometric points for {object_id}")
            return df

        except Exception as e:
            logger.warning(f"Failed to retrieve light curve for {object_id}: {e}")
            return None

    def get_stamps(self, object_id: str, ra: float, dec: float) -> Dict[str, Any]:
        """Retrieve postage stamps/thumbnails from ANTARES."""
        try:
            from antares_client.search import get_by_id

            locus = get_by_id(object_id)
            if locus is None:
                return {}

            stamps = {}
            if hasattr(locus, 'alerts') and locus.alerts:
                for alert in locus.alerts[-3:]:
                    props = alert.properties
                    for key in ['ztf_cutoutScience', 'ztf_cutoutTemplate',
                                'ztf_cutoutDifference']:
                        if key in props and props[key]:
                            stamps[key] = props[key]

            if stamps:
                return {
                    'stamps': stamps,
                    'ra': ra,
                    'dec': dec,
                    'object_id': object_id,
                    'broker': 'ANTARES',
                }
            return {}

        except Exception as e:
            logger.warning(f"Failed to retrieve stamps for {object_id}: {e}")
            return {}
