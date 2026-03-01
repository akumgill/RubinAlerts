"""Main monitoring pipeline for Type Ia supernovae in Rubin Deep Drilling Fields."""

import logging
from datetime import datetime
from typing import Optional, Dict, List
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from broker_clients.antares_client import AntaresClient
from broker_clients.alerce_client import AlerceClient
from broker_clients.atlas_client import AtlasClient
from broker_clients.rubin_tap_client import RubinTAPClient
from broker_clients.fink_client import FinkLSSTClient
from core.alert_aggregator import AlertAggregator
from core.variable_screen import VariableScreener
from core.ddf_fields import DDF_FIELDS
from host_galaxy.morphology_filter import MorphologyFilter
from cache.alert_cache import AlertCache
from utils.extinction import get_extinction_batch
from utils.ned_query import query_ned_batch


class SupernovaMonitor:
    """Main monitoring pipeline for SN Ia in Rubin DDFs."""

    def __init__(self, cache_dir: str = './cache/data',
                 use_alerce_db: bool = True,
                 apply_extinction: bool = True,
                 query_ned: bool = True,
                 use_rsp: bool = True):
        self.cache_dir = cache_dir
        self.cache = AlertCache(cache_dir)
        self.apply_extinction = apply_extinction
        self.query_ned = query_ned
        self.use_alerce_db = use_alerce_db
        self.use_rsp = use_rsp
        self.aggregator = AlertAggregator(
            cache_dir,
            apply_extinction=apply_extinction,
        )
        self.morphology_filter = MorphologyFilter(cache_dir)
        self.variable_screener = VariableScreener()
        self.atlas_client = AtlasClient()
        self._lc_cache = {}  # in-memory light curve cache: (broker, oid) -> DataFrame
        self._rsp_lc_cache = {}  # RSP light curves: object_id -> DataFrame

        # RSP TAP client for host galaxy properties (DP1/DR1 static catalogs)
        self.rsp_client = None
        if use_rsp:
            try:
                self.rsp_client = RubinTAPClient()
                if self.rsp_client.available:
                    logger.info("RSP TAP client initialized (token configured)")
                else:
                    logger.warning("RSP TAP client: token or pyvo not available")
                    self.rsp_client = None
            except Exception as e:
                logger.warning("Failed to initialize RSP TAP client: %s", e)
                self.rsp_client = None

        # Fink LSST client for current prompt photometry
        self.fink_client = None
        try:
            self.fink_client = FinkLSSTClient()
            if self.fink_client.available:
                logger.info("Fink LSST client initialized (API reachable)")
            else:
                logger.warning("Fink LSST API not reachable")
                self.fink_client = None
        except Exception as e:
            logger.warning("Failed to initialize Fink LSST client: %s", e)
            self.fink_client = None

        self.brokers = {}
        self._initialize_brokers()

    def _initialize_brokers(self):
        """Initialize broker clients."""
        try:
            self.brokers['ANTARES'] = AntaresClient(self.cache_dir)
            logger.info("ANTARES client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ANTARES: {e}")

        try:
            self.brokers['ALeRCE'] = AlerceClient(
                self.cache_dir, survey='ztf', use_db=self.use_alerce_db,
            )
            logger.info("ALeRCE (ZTF) client initialized (db=%s)", self.use_alerce_db)
        except Exception as e:
            logger.warning(f"Failed to initialize ALeRCE ZTF: {e}")

        try:
            self.brokers['ALeRCE-LSST'] = AlerceClient(
                self.cache_dir, survey='lsst', use_db=self.use_alerce_db,
            )
            logger.info("ALeRCE (LSST) client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ALeRCE LSST: {e}")

        if not self.brokers:
            logger.error("No broker clients could be initialized!")

    def query_all_brokers(self,
                         class_name: str = 'SNIa',
                         min_probability: float = 0.5,
                         days_back: int = 30,
                         limit: int = 200,
                         require_rubin: bool = False,
                         ddf_fields: Optional[List[Dict]] = None) -> Dict[str, pd.DataFrame]:
        """Query all available brokers, restricted to DDF fields.

        Each broker client manages its own JSON file cache (keyed by
        query parameters). The SQLite cache is used only for merged
        results and galaxy info, not for raw broker queries.
        """
        logger.info(f"Querying {len(self.brokers)} brokers for SN Ia candidates")

        results = {}

        for broker_name, client in self.brokers.items():
            logger.info(f"Querying {broker_name}...")

            try:
                kwargs = dict(
                    class_name=class_name,
                    min_probability=min_probability,
                    days_back=days_back,
                    limit=limit,
                    ddf_fields=ddf_fields,
                )
                if broker_name == 'ANTARES':
                    kwargs['require_rubin'] = require_rubin

                alerts = client.query_alerts(**kwargs)

                if alerts is not None and len(alerts) > 0:
                    results[broker_name] = alerts
                    logger.info(f"Retrieved {len(alerts)} alerts from {broker_name}")
                else:
                    logger.warning(f"No alerts from {broker_name}")

            except Exception as e:
                logger.error(f"Error querying {broker_name}: {e}")

        return results

    def run_full_pipeline(self,
                         min_ia_probability: float = 0.5,
                         days_back: int = 30,
                         limit: int = 200,
                         filter_elliptical: bool = False,
                         require_rubin: bool = True,
                         min_agreement: float = 0.0,
                         min_brokers: int = 1,
                         ddf_fields: Optional[List[Dict]] = None,
                         atlas_enrichment: bool = False) -> Optional[pd.DataFrame]:
        """
        Run complete monitoring pipeline.

        Args:
            min_ia_probability: Minimum Type Ia probability
            days_back: Query alerts from last N days
            limit: Max alerts per broker
            filter_elliptical: If True, only keep candidates in elliptical galaxies
            require_rubin: If True, only keep ANTARES loci with Rubin/LSST data
            min_agreement: Minimum agreement score (0.0 to skip)
            min_brokers: Minimum number of brokers detecting object
            ddf_fields: DDF fields to search (default: all 6 Rubin DDFs)
            atlas_enrichment: If True, fetch ATLAS forced photometry
        """
        fields = ddf_fields or DDF_FIELDS
        field_names = [f['name'] for f in fields]

        logger.info("=" * 60)
        logger.info("Starting SuperNova Monitoring Pipeline")
        logger.info(f"Time: {datetime.now().isoformat()}")
        logger.info(f"DDFs: {', '.join(field_names)}")
        logger.info(f"Parameters: P(Ia) >= {min_ia_probability}, "
                     f"days_back={days_back}, limit={limit}, "
                     f"require_rubin={require_rubin}, "
                     f"elliptical_filter={filter_elliptical}, "
                     f"atlas_enrichment={atlas_enrichment}")
        logger.info(f"Features: extinction={self.apply_extinction}, "
                     f"alerce_db={self.use_alerce_db}, "
                     f"ned_redshifts={self.query_ned}, "
                     f"rsp_photometry={self.rsp_client is not None}")
        logger.info("=" * 60)

        # Step 1: Query brokers (restricted to DDFs)
        alerts_by_broker = self.query_all_brokers(
            min_probability=min_ia_probability,
            days_back=days_back,
            limit=limit,
            require_rubin=require_rubin,
            ddf_fields=fields,
        )

        if not alerts_by_broker:
            logger.error("No alerts retrieved from any broker")
            return None

        for broker, df in alerts_by_broker.items():
            logger.info(f"  {broker}: {len(df)} raw alerts")

        # Step 2: Merge and deduplicate
        logger.info("Merging alerts from brokers...")
        merged_alerts = self.aggregator.merge_alerts(alerts_by_broker)

        if len(merged_alerts) == 0:
            logger.warning("No merged alerts after combining brokers")
            return None

        logger.info(f"Step 2: {len(merged_alerts)} unique objects after deduplication")

        # Step 3: Screen against known variable star catalogs
        logger.info("Screening against known variable catalogs...")
        merged_alerts = self.variable_screener.screen_candidates(merged_alerts)
        n_vars = merged_alerts['known_variable'].sum() if 'known_variable' in merged_alerts.columns else 0
        logger.info(f"Step 3: {n_vars} known variables flagged")

        # Step 4: Filter for high-confidence Type Ia (excludes known variables)
        logger.info("Filtering for high-confidence Type Ia candidates...")
        # Remove known variables before probability filtering
        candidates = merged_alerts.copy()
        if 'known_variable' in candidates.columns:
            candidates = candidates[~candidates['known_variable']]
            logger.info(f"  After removing known variables: {len(candidates)}")

        ia_candidates = self.aggregator.get_high_confidence_candidates(
            candidates,
            min_ia_probability=min_ia_probability,
            min_agreement=min_agreement,
            num_brokers=min_brokers
        )

        logger.info(f"Step 4: {len(ia_candidates)} candidates after probability filter")

        if len(ia_candidates) == 0:
            logger.warning("No candidates pass probability filter (P(Ia) >= %.2f). "
                           "Returning empty result. Lower min_ia_probability to see more.",
                           min_ia_probability)
            self._last_merged_alerts = merged_alerts
            return ia_candidates

        # Step 5: NED redshift lookups
        if self.query_ned and len(ia_candidates) > 0:
            logger.info("Querying NED for host galaxy redshifts...")
            try:
                ia_candidates = query_ned_batch(ia_candidates, cache=self.cache)
                n_z = ia_candidates['ned_redshift'].notna().sum()
                logger.info(f"Step 5: NED redshifts for {n_z}/{len(ia_candidates)} candidates")
            except Exception as e:
                logger.warning("NED redshift lookup failed: %s", e)

        # Step 6: Optionally filter for elliptical galaxies
        if filter_elliptical:
            logger.info("Filtering for elliptical galaxy hosts...")
            elliptical_candidates = self.morphology_filter.filter_elliptical(ia_candidates)
            logger.info(f"Step 6: {len(elliptical_candidates)} candidates in elliptical galaxies")

            if len(elliptical_candidates) == 0:
                logger.warning("No candidates in elliptical galaxies. "
                               "Continuing with all Type Ia candidates.")
            else:
                ia_candidates = elliptical_candidates

        # Step 7: Enrich with ATLAS forced photometry
        if atlas_enrichment and self.atlas_client.available:
            logger.info("Enriching with ATLAS forced photometry...")
            ia_candidates = self.atlas_client.enrich_candidates(ia_candidates)
            n_atlas = ia_candidates['atlas_has_data'].sum() if 'atlas_has_data' in ia_candidates.columns else 0
            logger.info(f"Step 7: {n_atlas}/{len(ia_candidates)} candidates have ATLAS data")
        elif atlas_enrichment:
            logger.warning("ATLAS credentials not configured; skipping enrichment")

        # Step 8: Cross-match to Fink for current prompt photometry
        if self.fink_client is not None and len(ia_candidates) > 0:
            logger.info("Cross-matching candidates to Fink LSST...")
            try:
                ia_candidates = self.fink_client.crossmatch_candidates(
                    ia_candidates
                )
                n_fink = ia_candidates['fink_diaObjectId'].notna().sum()
                logger.info(f"Step 8: {n_fink}/{len(ia_candidates)} candidates "
                            "matched in Fink (current prompt photometry)")
            except Exception as e:
                logger.warning("Fink cross-match failed: %s", e)

        # Step 8b: Cross-match to RSP for host galaxy properties (static catalogs)
        if self.rsp_client is not None and len(ia_candidates) > 0:
            logger.info("Cross-matching candidates to RSP DiaObjects...")
            try:
                cov = self.rsp_client.check_data_coverage()
                logger.info(f"  RSP {cov['schema']}: {cov['min_date']} to "
                            f"{cov['max_date']} ({cov['staleness_days']}d ago), "
                            f"{cov['n_dia_objects']:,} DiaObjects")
                ia_candidates = self.rsp_client.crossmatch_to_dia_objects(
                    ia_candidates
                )
                n_rsp = ia_candidates['diaObjectId'].notna().sum()
                logger.info(f"Step 8b: {n_rsp}/{len(ia_candidates)} candidates "
                            "matched to RSP DiaObjects (host properties)")
            except Exception as e:
                logger.warning("RSP cross-match failed: %s", e)

        logger.info("=" * 60)
        logger.info(f"FINAL: {len(ia_candidates)} Type Ia candidates")
        logger.info("=" * 60)

        self._last_merged_alerts = merged_alerts
        return ia_candidates

    def get_light_curve(self, object_id: str, broker: str = 'ANTARES') -> Optional[pd.DataFrame]:
        """Retrieve light curve for an object from a broker (cached)."""
        if broker not in self.brokers:
            logger.error(f"Broker {broker} not available")
            return None
        key = (broker, str(object_id))
        if key in self._lc_cache:
            logger.info(f"Light curve cache hit for {object_id} ({broker})")
            return self._lc_cache[key]
        lc = self.brokers[broker].get_light_curve(object_id)
        self._lc_cache[key] = lc
        return lc

    def get_fink_light_curve(self, object_id: str = None,
                             fink_dia_object_id: str = None,
                             ra: float = None, dec: float = None,
                             include_forced: bool = True) -> Optional[pd.DataFrame]:
        """Retrieve current Rubin photometry from Fink.

        This is the primary method for up-to-date light curves.
        Provide fink_dia_object_id directly, or (ra, dec) for
        positional matching.
        """
        if self.fink_client is None:
            logger.warning("Fink LSST client not available")
            return None

        # Resolve Fink diaObjectId
        if fink_dia_object_id is None and ra is not None and dec is not None:
            result = self.fink_client.cone_search(ra, dec, radius_arcsec=2.0)
            if result is not None and len(result) > 0:
                fink_dia_object_id = str(result.iloc[0].get("r:diaObjectId", ""))

        if not fink_dia_object_id:
            logger.warning("Could not resolve Fink diaObjectId")
            return None

        return self.fink_client.get_light_curve(
            fink_dia_object_id, include_forced=include_forced
        )

    def get_rsp_light_curve(self, object_id: str = None,
                            dia_object_id: int = None,
                            ra: float = None, dec: float = None,
                            include_forced: bool = True) -> Optional[pd.DataFrame]:
        """Retrieve light curve from RSP for a candidate.

        Provide either dia_object_id directly, or object_id (to look up
        the cached diaObjectId from cross-matching), or (ra, dec) for an
        on-the-fly positional match.
        """
        if self.rsp_client is None:
            logger.warning("RSP TAP client not available")
            return None

        # Resolve diaObjectId
        if dia_object_id is None and object_id is not None:
            # Check RSP light curve cache
            if object_id in self._rsp_lc_cache:
                return self._rsp_lc_cache[object_id]

        if dia_object_id is None and ra is not None and dec is not None:
            # On-the-fly cross-match
            tmp = pd.DataFrame({'ra': [ra], 'dec': [dec]})
            matched = self.rsp_client.crossmatch_to_dia_objects(tmp)
            if matched['diaObjectId'].notna().any():
                dia_object_id = int(matched.iloc[0]['diaObjectId'])

        if dia_object_id is None:
            logger.warning("Could not resolve diaObjectId for RSP query")
            return None

        try:
            lc = self.rsp_client.get_light_curve(
                dia_object_id, include_forced=include_forced
            )
            if object_id is not None:
                self._rsp_lc_cache[object_id] = lc
            return lc
        except Exception as e:
            logger.warning("Failed to get RSP light curve: %s", e)
            return None

    def get_atlas_light_curve(self, ra: float, dec: float,
                              mjd_min: Optional[float] = None) -> Optional[pd.DataFrame]:
        """Retrieve ATLAS forced photometry light curve for a coordinate."""
        if not self.atlas_client.available:
            logger.warning("ATLAS credentials not configured")
            return None
        return self.atlas_client.get_light_curve(ra, dec, mjd_min=mjd_min)

    def get_stamps(self, object_id: str, ra: float, dec: float,
                   broker: str = 'ANTARES') -> Dict:
        """Retrieve postage stamps for an object."""
        if broker not in self.brokers:
            logger.error(f"Broker {broker} not available")
            return {}
        return self.brokers[broker].get_stamps(object_id, ra, dec)
