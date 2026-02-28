"""ANTARES alert broker client — DDF-focused transient search."""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from .base_client import BaseBrokerClient
from core.ddf_fields import DDF_FIELDS, DDF_SEARCH_RADIUS_DEG

logger = logging.getLogger(__name__)

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

    def query_alerts(self,
                    class_name: str = 'SN Ia',
                    min_probability: float = 0.5,
                    days_back: int = 30,
                    limit: int = 200,
                    require_rubin: bool = False,
                    ddf_fields: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Query ANTARES for transient candidates in Rubin Deep Drilling Fields.

        Strategy:
        1. Cone search each DDF for loci
        2. Filter for galaxy-associated objects (sgscore < 0.5)
        3. Filter for transient-like duration (< 200 days)
        4. Filter for real detections (rb >= 0.5)
        5. Optionally require Rubin/LSST photometry
        """
        fields = ddf_fields or DDF_FIELDS
        field_names = ','.join(f['name'] for f in fields)
        cache_key = f"alerts_v4_ddf_{field_names}_d{days_back}_n{limit}_rubin{int(require_rubin)}"
        cache_path = self._get_cache_path(cache_key)
        cached_data = self._load_cache(cache_path)
        if cached_data:
            return pd.DataFrame(cached_data)

        try:
            from antares_client.search import search, cone_search
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            alerts_list = []
            seen_ids = set()
            total_checked = 0

            # Allocate quota per field so all DDFs get searched
            per_field_limit = max(limit // len(fields), 20)
            max_checked_per_field = 500  # don't iterate forever

            # Search each DDF field
            for field in fields:
                field_name = field['name']
                center = SkyCoord(ra=field['ra'] * u.deg, dec=field['dec'] * u.deg)
                radius = DDF_SEARCH_RADIUS_DEG * u.deg

                logger.info(f"  Searching DDF {field_name} "
                           f"(RA={field['ra']:.2f}, Dec={field['dec']:.2f}, "
                           f"r={DDF_SEARCH_RADIUS_DEG:.2f} deg)")

                field_count = 0
                field_accepted = 0

                try:
                    for locus in cone_search(center, radius):
                        if locus.locus_id in seen_ids:
                            continue
                        seen_ids.add(locus.locus_id)
                        total_checked += 1
                        field_count += 1

                        # Cap check MUST be outside inner try so `continue` can't skip it
                        if field_accepted >= per_field_limit:
                            break
                        if field_count >= max_checked_per_field:
                            logger.info(f"    {field_name}: hit search cap ({max_checked_per_field})")
                            break

                        if field_count % 200 == 0:
                            logger.info(f"    {field_name}: checked {field_count} loci, "
                                       f"accepted {field_accepted} so far...")

                        try:
                            alert_dict = self._parse_locus(locus, f'ddf_{field_name}')
                            alert_dict['ddf_field'] = field_name
                            alert_dict['has_rubin'] = self._locus_has_rubin_data(locus)
                            alert_dict['has_ztf'] = self._locus_has_ztf_data(locus)

                            # Quality filters
                            if not self._passes_quality_cuts(alert_dict):
                                continue

                            if require_rubin and not alert_dict['has_rubin']:
                                continue

                            alerts_list.append(alert_dict)
                            field_accepted += 1
                        except Exception as e:
                            logger.debug(f"Failed to parse locus {locus.locus_id}: {e}")
                except Exception as e:
                    logger.warning(f"Error searching DDF {field_name}: {e}")

                logger.info(f"    {field_name}: checked {field_count}, "
                           f"accepted {field_accepted}")

            logger.info(f"DDF search: checked {total_checked} loci total, "
                       f"accepted {len(alerts_list)} transient candidates")

            if alerts_list:
                self._save_cache(cache_path, alerts_list)

            df = pd.DataFrame(alerts_list)
            logger.info(f"Retrieved {len(df)} ANTARES alerts from DDFs")
            return df

        except Exception as e:
            logger.error(f"Error querying ANTARES: {e}")
            return pd.DataFrame()

    def _passes_quality_cuts(self, alert_dict: Dict) -> bool:
        """Check if a locus passes quality cuts for transient candidacy."""
        # Star/galaxy separation: reject point sources
        sgscore = alert_dict.get('ztf_sgscore1')
        if sgscore is not None and sgscore >= MAX_SGSCORE:
            return False

        # Duration: reject persistent variables
        duration = alert_dict.get('duration_days')
        if duration is not None and duration > MAX_DURATION_DAYS:
            return False

        # Real/bogus: reject likely artifacts
        rb = alert_dict.get('ztf_rb')
        if rb is not None and rb < MIN_RB_SCORE:
            return False

        return True

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
