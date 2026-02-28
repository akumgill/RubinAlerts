"""ALeRCE alert broker client — DDF-focused SN Ia search.

Supports both ZTF and LSST surveys via the survey parameter.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from .base_client import BaseBrokerClient
from .alerce_db_client import AlerceDBClient
from core.ddf_fields import DDF_FIELDS, DDF_SEARCH_RADIUS_DEG

logger = logging.getLogger(__name__)

# ZTF classifier configuration
ZTF_CLASSIFIER_PRIMARY = "lc_classifier_BHRF_forced_phot"
ZTF_CLASSIFIER_FALLBACK = "lc_classifier_transient"
ZTF_CLASS_NAME = "SNIa"

# LSST classifier configuration
LSST_CLASSIFIER = "stamp_classifier_rubin"
LSST_CLASS_NAME = "SN"

# LSST band code → name mapping
LSST_BAND_NAMES = {1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y', 6: 'u'}


class AlerceClient(BaseBrokerClient):
    """Client for querying ALeRCE alert broker (alerce 2.2.1).

    Supports two survey modes:
    - survey='ztf': Uses lc_classifier_BHRF_forced_phot (v2.1.0) for SN Ia
      classification via forced photometry. Falls back to lc_classifier_transient.
    - survey='lsst': Uses stamp_classifier_rubin for SN classification
      from Rubin/LSST alert stream data.
    """

    def __init__(self, cache_dir: str = './cache/data', survey: str = 'ztf',
                 use_db: bool = True):
        broker_name = 'ALeRCE' if survey == 'ztf' else 'ALeRCE-LSST'
        super().__init__(broker_name, cache_dir)
        self.survey = survey
        self.alerce = None
        self.use_db = use_db
        self.db_client = AlerceDBClient() if use_db else None
        self._initialize_client()
        os.makedirs(cache_dir, exist_ok=True)

    def _initialize_client(self):
        """Initialize ALeRCE client library."""
        try:
            from alerce.core import Alerce
            self.alerce = Alerce()
            logger.info(f"ALeRCE client initialized successfully (survey={self.survey})")
        except ImportError:
            logger.warning("alerce library not available. Install with: pip3 install alerce")
            raise

    def _get_cache_path(self, query_type: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.cache_dir, f"alerce_{query_type}_{timestamp}.json")

    def _load_cache(self, cache_path: str) -> Optional[list]:
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded ALeRCE cache from {cache_path}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_cache(self, cache_path: str, data: list):
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, default=str)
                logger.info(f"Saved ALeRCE cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    # ------------------------------------------------------------------
    # Alert queries
    # ------------------------------------------------------------------

    def query_alerts(self,
                    class_name: str = 'SNIa',
                    min_probability: float = 0.5,
                    days_back: int = 30,
                    limit: int = 200,
                    ddf_fields: Optional[List[Dict]] = None,
                    **kwargs) -> pd.DataFrame:
        """Query ALeRCE for transient candidates in Rubin DDFs.

        Dispatches to the ZTF or LSST query path based on self.survey.
        """
        if self.alerce is None:
            logger.error("ALeRCE client not initialized")
            return pd.DataFrame()

        if self.survey == 'lsst':
            return self._query_alerts_lsst(
                min_probability=min_probability,
                days_back=days_back,
                limit=limit,
                ddf_fields=ddf_fields,
            )
        else:
            # Try direct DB access for ZTF (much faster for bulk queries)
            if self.use_db and self.db_client and self.db_client.available:
                try:
                    return self._query_alerts_via_db(
                        class_name=class_name,
                        min_probability=min_probability,
                        days_back=days_back,
                        limit=limit,
                        ddf_fields=ddf_fields,
                    )
                except Exception as e:
                    logger.warning("ALeRCE DB query failed, falling back to REST API: %s", e)

            return self._query_alerts_ztf(
                class_name=class_name,
                min_probability=min_probability,
                days_back=days_back,
                limit=limit,
                ddf_fields=ddf_fields,
            )

    # --- ZTF query via direct database ---

    def _query_alerts_via_db(self,
                             class_name: str = 'SNIa',
                             min_probability: float = 0.5,
                             days_back: int = 30,
                             limit: int = 200,
                             ddf_fields: Optional[List[Dict]] = None) -> pd.DataFrame:
        """Query ALeRCE ZTF via direct PostgreSQL database access.

        Faster than REST API for bulk queries. Falls back to REST on failure.
        """
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astropy.time import Time

        logger.info("Querying ALeRCE via direct database access...")
        self.db_client.connect()

        # Get SN candidates from DB
        candidates = self.db_client.query_sn_candidates(
            min_prob=min_probability,
            classifier='lc_classifier',
        )

        if len(candidates) == 0:
            logger.info("No SN candidates from ALeRCE DB")
            return pd.DataFrame()

        # Get unique objects with their top classification
        top = candidates[candidates['ranking'] == 1].copy()
        if len(top) == 0:
            top = candidates.drop_duplicates('oid', keep='first')

        # Filter by DDF field coordinates
        fields = ddf_fields or DDF_FIELDS
        obj_coords = SkyCoord(
            ra=top['meanra'].values * u.deg,
            dec=top['meandec'].values * u.deg,
        )

        in_any_ddf = np.zeros(len(top), dtype=bool)
        ddf_assignments = [''] * len(top)

        for field in fields:
            center = SkyCoord(ra=field['ra'] * u.deg, dec=field['dec'] * u.deg)
            seps = obj_coords.separation(center)
            in_field = seps <= DDF_SEARCH_RADIUS_DEG * u.deg
            for idx_pos in np.where(in_field)[0]:
                if not in_any_ddf[idx_pos]:
                    ddf_assignments[idx_pos] = field['name']
            in_any_ddf |= in_field

        ddf_top = top[in_any_ddf].copy()
        ddf_names = [ddf_assignments[i] for i in np.where(in_any_ddf)[0]]

        logger.info("ALeRCE DB: %d/%d candidates in DDFs", len(ddf_top), len(top))

        if len(ddf_top) == 0:
            return pd.DataFrame()

        # Limit results
        if len(ddf_top) > limit:
            ddf_top = ddf_top.head(limit)
            ddf_names = ddf_names[:limit]

        # Get full probabilities for DDF candidates
        oids = ddf_top['oid'].tolist()
        prob_df = self.db_client.query_probabilities(oids)

        # Get PS1 host data
        ps1_df = self.db_client.query_ps1_host(oids)

        # Build standard alert format
        alerts_list = []
        prob_lookup = {}
        if len(prob_df) > 0:
            prob_lookup = prob_df.set_index('oid').to_dict('index')

        ps1_lookup = {}
        if len(ps1_df) > 0:
            ps1_lookup = ps1_df.set_index('oid').to_dict('index')

        for (_, row), ddf_name in zip(ddf_top.iterrows(), ddf_names):
            oid = row['oid']
            alert = {
                'object_id': oid,
                'ra': row['meanra'],
                'dec': row['meandec'],
                'discovery_date': row.get('firstMJD'),
                'last_mjd': None,
                'n_detections': row.get('ndet'),
                'sn_ia_prob': row.get('probability'),
                'classifier': row.get('classifier_name', 'lc_classifier'),
                'class': row.get('class_name', class_name),
                'stellar': False,
                'magnitude': row.get('g_r_max'),
                'broker': 'ALeRCE',
                'alerce_survey': 'ztf',
                'ddf_field': ddf_name,
            }

            # Add all class probabilities
            probs = prob_lookup.get(oid, {})
            for key, val in probs.items():
                if key.startswith('prob_') and pd.notna(val):
                    alert[key] = float(val)

            # Add PS1 host data
            ps1 = ps1_lookup.get(oid, {})
            if ps1:
                alert['ps1_sgscore'] = ps1.get('sgscore1')
                alert['ps1_g_r_host'] = ps1.get('g_r_host')
                alert['ps1_r_i_host'] = ps1.get('r_i_host')

            alerts_list.append(alert)

        df = pd.DataFrame(alerts_list)
        logger.info("ALeRCE DB: Retrieved %d alerts from DDFs", len(df))

        if len(df) > 0 and 'ddf_field' in df.columns:
            for field_name, count in df['ddf_field'].value_counts().items():
                logger.info("    %s: %d", field_name, count)

        return df

    # --- ZTF query path via REST API ---

    def _query_alerts_ztf(self,
                          class_name: str = 'SNIa',
                          min_probability: float = 0.5,
                          days_back: int = 30,
                          limit: int = 200,
                          ddf_fields: Optional[List[Dict]] = None) -> pd.DataFrame:
        """Query ALeRCE ZTF survey for SN Ia candidates in DDFs."""
        fields = ddf_fields or DDF_FIELDS
        field_names = ','.join(f['name'] for f in fields)
        cache_key = (f"alerts_v3_{ZTF_CLASSIFIER_PRIMARY}_ddf_{field_names}"
                     f"_p{min_probability}_d{days_back}_n{limit}")
        cache_path = self._get_cache_path(cache_key)
        cached_data = self._load_cache(cache_path)
        if cached_data:
            return pd.DataFrame(cached_data)

        try:
            from astropy.time import Time
            cutoff_date = datetime.now() - timedelta(days=days_back)
            cutoff_mjd = Time(cutoff_date.isoformat(), format='isot').mjd

            all_alerts = []
            seen_oids = set()

            for field in fields:
                if len(all_alerts) >= limit:
                    break

                field_name = field['name']
                logger.info(f"  Querying ALeRCE ZTF in DDF {field_name}")

                field_alerts = self._query_ddf_field_ztf(
                    field, class_name, min_probability,
                    cutoff_mjd, limit - len(all_alerts),
                )

                for alert in field_alerts:
                    oid = alert.get('object_id')
                    if oid and oid not in seen_oids:
                        seen_oids.add(oid)
                        alert['ddf_field'] = field_name
                        all_alerts.append(alert)

                logger.info(f"    {field_name}: {len(field_alerts)} candidates")

            all_alerts = self._enrich_with_all_probabilities(all_alerts)

            if all_alerts:
                self._save_cache(cache_path, all_alerts)

            df = pd.DataFrame(all_alerts)
            logger.info(f"Retrieved {len(df)} ALeRCE ZTF alerts from DDFs")
            return df

        except Exception as e:
            logger.error(f"Error querying ALeRCE ZTF: {e}")
            return pd.DataFrame()

    def _query_ddf_field_ztf(self, field: Dict, class_name: str,
                              min_probability: float, cutoff_mjd: float,
                              remaining: int) -> List[Dict]:
        """Query a single DDF field via ZTF, trying primary then fallback classifier."""
        for classifier in [ZTF_CLASSIFIER_PRIMARY, ZTF_CLASSIFIER_FALLBACK]:
            alerts = self._query_classifier_in_field_ztf(
                field, classifier, class_name,
                min_probability, cutoff_mjd, remaining,
            )
            if alerts:
                return alerts
            if classifier == ZTF_CLASSIFIER_PRIMARY:
                logger.info(f"    No results from {classifier}, trying fallback")
        return []

    def _query_classifier_in_field_ztf(self, field: Dict, classifier: str,
                                        class_name: str, min_probability: float,
                                        cutoff_mjd: float, remaining: int) -> List[Dict]:
        """Query a single ZTF classifier in a single DDF field with pagination."""
        alerts_list = []
        page = 1
        page_size = min(remaining, 100)

        while len(alerts_list) < remaining:
            try:
                results = self.alerce.query_objects(
                    survey='ztf',
                    classifier=classifier,
                    class_name=class_name,
                    probability=min_probability,
                    ra=field['ra'],
                    dec=field['dec'],
                    radius=DDF_SEARCH_RADIUS_DEG,
                    ndet_min=5,
                    format="pandas",
                    page_size=page_size,
                    page=page,
                )
            except Exception as e:
                logger.debug(f"ALeRCE ZTF query error (page {page}): {e}")
                break

            if results is None or len(results) == 0:
                break

            for _, row in results.iterrows():
                alerts_list.append({
                    'object_id': row.get('oid'),
                    'ra': row.get('meanra'),
                    'dec': row.get('meandec'),
                    'discovery_date': row.get('firstmjd'),
                    'last_mjd': row.get('lastmjd'),
                    'n_detections': row.get('ndet'),
                    'sn_ia_prob': row.get('probability'),
                    'classifier': classifier,
                    'class': row.get('class'),
                    'stellar': row.get('stellar'),
                    'magnitude': row.get('g_r_max'),
                    'broker': 'ALeRCE',
                    'alerce_survey': 'ztf',
                })

            if len(results) < page_size:
                break
            page += 1

        return alerts_list

    # --- LSST query path ---

    def _query_alerts_lsst(self,
                           min_probability: float = 0.5,
                           days_back: int = 30,
                           limit: int = 200,
                           ddf_fields: Optional[List[Dict]] = None) -> pd.DataFrame:
        """Query ALeRCE LSST survey for SN candidates in DDFs.

        The LSST cone search API is not operational, so we query globally
        for all SN-classified objects and filter by DDF coordinates locally.
        """
        fields = ddf_fields or DDF_FIELDS
        field_names = ','.join(f['name'] for f in fields)
        cache_key = (f"lsst_alerts_v1_{LSST_CLASSIFIER}_ddf_{field_names}"
                     f"_p{min_probability}_d{days_back}_n{limit}")
        cache_path = self._get_cache_path(cache_key)
        cached_data = self._load_cache(cache_path)
        if cached_data:
            return pd.DataFrame(cached_data)

        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            from astropy.time import Time

            cutoff_date = datetime.now() - timedelta(days=days_back)
            cutoff_mjd = Time(cutoff_date.isoformat(), format='isot').mjd

            # Query LSST SN objects globally (cone search is broken on API).
            # Cap at max_pages to keep runtime reasonable — the API returns
            # objects sorted by descending probability, so we get the best
            # candidates first.
            max_pages = max(1, limit // 100) + 1  # e.g. limit=1000 → 11 pages
            logger.info(f"  Querying ALeRCE LSST globally for {LSST_CLASS_NAME} "
                        f"(classifier={LSST_CLASSIFIER}, max_pages={max_pages})")

            all_results = []
            page = 1
            page_size = 100

            while page <= max_pages:
                try:
                    results = self.alerce.query_objects(
                        survey='lsst',
                        classifier=LSST_CLASSIFIER,
                        class_name=LSST_CLASS_NAME,
                        format="pandas",
                        page_size=page_size,
                        page=page,
                    )
                except Exception as e:
                    logger.debug(f"ALeRCE LSST query error (page {page}): {e}")
                    break

                if results is None or len(results) == 0:
                    break

                all_results.append(results)
                logger.info(f"    Page {page}: {len(results)} objects")

                if len(results) < page_size:
                    break
                page += 1

            if not all_results:
                logger.info("  No LSST SN objects found in ALeRCE")
                return pd.DataFrame()

            all_objects = pd.concat(all_results, ignore_index=True)
            logger.info(f"  Total LSST objects retrieved: {len(all_objects)}")

            # Filter by DDF coordinates locally
            obj_coords = SkyCoord(
                ra=all_objects['meanra'].values * u.deg,
                dec=all_objects['meandec'].values * u.deg,
            )

            in_any_ddf = np.zeros(len(all_objects), dtype=bool)
            ddf_assignments = [''] * len(all_objects)

            for field in fields:
                center = SkyCoord(ra=field['ra'] * u.deg, dec=field['dec'] * u.deg)
                seps = obj_coords.separation(center)
                in_field = seps <= DDF_SEARCH_RADIUS_DEG * u.deg
                for idx in np.where(in_field)[0]:
                    if not in_any_ddf[idx]:
                        ddf_assignments[idx] = field['name']
                in_any_ddf |= in_field

            ddf_objects = all_objects[in_any_ddf].copy()
            ddf_names = [ddf_assignments[i] for i in np.where(in_any_ddf)[0]]

            logger.info(f"  {len(ddf_objects)} objects in DDFs "
                        f"(out of {len(all_objects)} total)")

            if len(ddf_objects) == 0:
                return pd.DataFrame()

            # Convert to standard alert format
            alerts_list = []
            for (_, row), ddf_name in zip(ddf_objects.iterrows(), ddf_names):
                oid = row.get('oid')
                alerts_list.append({
                    'object_id': str(oid),
                    'ra': row.get('meanra'),
                    'dec': row.get('meandec'),
                    'discovery_date': row.get('firstmjd'),
                    'last_mjd': row.get('lastmjd'),
                    'n_detections': row.get('n_det'),
                    'n_forced': row.get('n_forced', 0),
                    'sn_ia_prob': row.get('probability'),
                    'classifier': LSST_CLASSIFIER,
                    'class': row.get('class_name', LSST_CLASS_NAME),
                    'stellar': row.get('stellar'),
                    'magnitude': None,
                    'broker': 'ALeRCE-LSST',
                    'alerce_survey': 'lsst',
                    'ddf_field': ddf_name,
                })

            # Skip per-object probability enrichment for LSST — the stamp
            # classifier probability is already in sn_ia_prob from query_objects,
            # and enriching hundreds of objects one-by-one is too slow.

            if alerts_list:
                self._save_cache(cache_path, alerts_list)

            df = pd.DataFrame(alerts_list)
            logger.info(f"Retrieved {len(df)} ALeRCE LSST alerts from DDFs")

            # Log per-DDF breakdown
            if len(df) > 0 and 'ddf_field' in df.columns:
                for field_name, count in df['ddf_field'].value_counts().items():
                    logger.info(f"    {field_name}: {count}")

            return df

        except Exception as e:
            logger.error(f"Error querying ALeRCE LSST: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Probability enrichment
    # ------------------------------------------------------------------

    def _enrich_with_all_probabilities(self, alerts_list: list) -> list:
        """Fetch full classification probabilities for each object."""
        for alert in alerts_list:
            oid = alert.get('object_id')
            if not oid:
                continue
            try:
                survey = alert.get('alerce_survey', self.survey)
                probs = self.alerce.query_probabilities(
                    oid=oid,
                    survey=survey,
                    format="pandas",
                )
                if probs is not None and len(probs) > 0:
                    if survey == 'lsst':
                        # LSST stamp classifier: extract all class probs
                        for _, prow in probs.iterrows():
                            cls = prow.get('class_name', '')
                            prob = prow.get('probability', 0)
                            key = f"prob_{cls}".lower().replace(' ', '_').replace('/', '_')
                            alert[key] = prob
                    else:
                        # ZTF: filter to transient classifiers
                        transient = probs[probs['classifier_name'].isin([
                            ZTF_CLASSIFIER_PRIMARY,
                            ZTF_CLASSIFIER_FALLBACK,
                        ])]
                        for _, prow in transient.iterrows():
                            cls = prow.get('class_name', '')
                            prob = prow.get('probability', 0)
                            key = f"prob_{cls}".lower().replace(' ', '_').replace('/', '_')
                            alert[key] = prob
            except Exception as e:
                logger.debug(f"Could not get probabilities for {oid}: {e}")
        return alerts_list

    # ------------------------------------------------------------------
    # Light curves
    # ------------------------------------------------------------------

    def get_light_curve(self, object_id: str) -> Optional[pd.DataFrame]:
        """Retrieve light curve from ALeRCE.

        Dispatches to ZTF or LSST path based on self.survey.
        """
        if self.alerce is None:
            return None

        if self.survey == 'lsst':
            return self._get_light_curve_lsst(object_id)
        else:
            return self._get_light_curve_ztf(object_id)

    def _get_light_curve_ztf(self, object_id: str) -> Optional[pd.DataFrame]:
        """Retrieve ZTF light curve from ALeRCE (existing behavior)."""
        try:
            logger.info(f"Retrieving ZTF light curve for ALeRCE object {object_id}")

            detections = self.alerce.query_detections(
                oid=object_id,
                survey='ztf',
                format="pandas",
            )

            if detections is not None and len(detections) > 0:
                col_map = {
                    'mjd': 'mjd',
                    'mag': 'magnitude',
                    'mag_corr': 'magnitude',
                    'magerr': 'mag_err',
                    'e_mag': 'mag_err',
                    'e_mag_corr': 'mag_err',
                    'fid': 'band',
                }
                df = detections.rename(columns={
                    k: v for k, v in col_map.items() if k in detections.columns
                })
                logger.info(f"Retrieved {len(df)} ZTF detections")
                return df

            return None

        except Exception as e:
            logger.warning(f"Failed to retrieve ZTF light curve for {object_id}: {e}")
            return None

    def _get_light_curve_lsst(self, object_id: str) -> Optional[pd.DataFrame]:
        """Retrieve LSST light curve from ALeRCE.

        Fetches both detections and forced photometry, converts psfFlux (nJy)
        to AB magnitudes, and combines into a single DataFrame.
        """
        try:
            logger.info(f"Retrieving LSST light curve for ALeRCE object {object_id}")

            frames = []

            # Fetch detections
            try:
                detections = self.alerce.query_detections(
                    oid=object_id,
                    survey='lsst',
                    format="pandas",
                )
                if detections is not None and len(detections) > 0:
                    det_lc = self._convert_lsst_flux_to_mag(detections, source='detection')
                    if det_lc is not None and len(det_lc) > 0:
                        frames.append(det_lc)
                        logger.info(f"  Detections: {len(det_lc)} points")
            except Exception as e:
                logger.debug(f"Failed to get LSST detections for {object_id}: {e}")

            # Fetch forced photometry
            try:
                forced = self.alerce.query_forced_photometry(
                    oid=object_id,
                    survey='lsst',
                    format="pandas",
                )
                if forced is not None and len(forced) > 0:
                    fp_lc = self._convert_lsst_flux_to_mag(forced, source='forced_phot')
                    if fp_lc is not None and len(fp_lc) > 0:
                        frames.append(fp_lc)
                        logger.info(f"  Forced photometry: {len(fp_lc)} points")
            except Exception as e:
                logger.debug(f"Failed to get LSST forced photometry for {object_id}: {e}")

            if not frames:
                logger.warning(f"No LSST light curve data for {object_id}")
                return None

            combined = pd.concat(frames, ignore_index=True)
            combined = combined.sort_values('mjd').reset_index(drop=True)

            logger.info(f"Retrieved {len(combined)} total LSST photometric points "
                        f"for {object_id}")

            if 'band' in combined.columns:
                for band, n in combined['band'].value_counts().items():
                    logger.info(f"    {band}: {n} pts")

            return combined

        except Exception as e:
            logger.warning(f"Failed to retrieve LSST light curve for {object_id}: {e}")
            return None

    @staticmethod
    def _convert_lsst_flux_to_mag(df: pd.DataFrame, source: str = 'detection') -> Optional[pd.DataFrame]:
        """Convert LSST psfFlux (nJy) to AB magnitudes, keeping ALL epochs.

        AB magnitude: mag = -2.5 * log10(flux_nJy) + 31.4
        Error propagation: mag_err = (2.5 / ln(10)) * (fluxErr / flux)

        Negative/zero flux points get magnitude=NaN but are still included
        with their raw flux values preserved — these are valid difference-imaging
        measurements where the transient is fainter than the template.
        """
        if 'psfFlux' not in df.columns or 'mjd' not in df.columns:
            return None

        flux = df['psfFlux'].values.astype(float)
        flux_err = df.get('psfFluxErr', pd.Series(np.zeros(len(df)))).values.astype(float)

        # Convert positive flux to magnitudes; negative flux → NaN magnitude
        valid = flux > 0
        magnitude = np.full(len(df), np.nan)
        mag_err = np.full(len(df), np.nan)

        magnitude[valid] = -2.5 * np.log10(flux[valid]) + 31.4
        mag_err[valid] = (2.5 / np.log(10)) * (flux_err[valid] / flux[valid])

        # Resolve band names: prefer band_name column, fall back to code mapping
        if 'band_name' in df.columns:
            bands = df['band_name'].values
        elif 'band' in df.columns:
            bands = df['band'].map(LSST_BAND_NAMES).fillna('unknown').values
        else:
            bands = np.full(len(df), 'unknown')

        result = pd.DataFrame({
            'mjd': df['mjd'].values,
            'magnitude': magnitude,
            'mag_err': mag_err,
            'flux': flux,
            'flux_err': flux_err,
            'band': bands,
            'survey': 'Rubin',
            'source': source,
        })

        # Also include scienceFlux (total flux from science image) when available
        if 'scienceFlux' in df.columns:
            result['science_flux'] = df['scienceFlux'].values.astype(float)
        if 'scienceFluxErr' in df.columns:
            result['science_flux_err'] = df['scienceFluxErr'].values.astype(float)

        # Keep ALL rows — don't drop negative flux points
        return result

    # ------------------------------------------------------------------
    # Postage stamps
    # ------------------------------------------------------------------

    def get_stamps(self, object_id: str, ra: float, dec: float) -> Dict[str, Any]:
        """Retrieve postage stamps from ALeRCE."""
        if self.alerce is None:
            return {}

        try:
            stamps = self.alerce.get_stamps(oid=object_id)

            if stamps:
                return {
                    'stamps': stamps,
                    'ra': ra,
                    'dec': dec,
                    'object_id': object_id,
                    'broker': self.broker_name,
                }
            return {}

        except Exception as e:
            logger.warning(f"Failed to retrieve stamps for {object_id}: {e}")
            return {}
