"""Alert aggregation and deduplication across multiple brokers."""

import logging
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

from utils.coordinates import CoordinateUtils
from utils.extinction import get_extinction_batch
from cache.alert_cache import AlertCache

logger = logging.getLogger(__name__)

# ANTARES tags — these indicate a classifier PROCESSED the object,
# NOT that the object IS a SN Ia. Used only as weak signals in the proxy.
ANTARES_PROCESSING_TAGS = {'superphot_plus_classified', 'SN_candies'}
ANTARES_TRANSIENT_TAGS = {'high_amplitude_transient_candidate'}


class AlertAggregator:
    """Aggregate and merge alerts from multiple brokers."""

    def __init__(self, cache_dir: str = './cache/data', match_tolerance_arcsec: float = 1.0,
                 apply_extinction: bool = True):
        self.cache = AlertCache(cache_dir)
        self.match_tolerance = match_tolerance_arcsec
        self.apply_extinction = apply_extinction

    def merge_alerts(self, alerts_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge alerts from multiple brokers into a deduplicated DataFrame."""
        if not alerts_dict:
            logger.warning("No alerts provided for merging")
            return pd.DataFrame()

        all_alerts = []
        for broker, df in alerts_dict.items():
            if df is not None and len(df) > 0:
                df = df.copy()
                df['broker_source'] = broker
                # Normalize probability columns before merging
                df = self._normalize_broker_probs(df, broker)
                all_alerts.append(df)

        if not all_alerts:
            logger.warning("No valid alert data to merge")
            return pd.DataFrame()

        combined_df = pd.concat(all_alerts, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} total alerts from {len(all_alerts)} brokers")

        merged_df = self._deduplicate_by_coordinates(combined_df)
        merged_df = self._add_classification_columns(merged_df)

        # Galactic extinction lookup
        if self.apply_extinction and len(merged_df) > 0:
            try:
                logger.info("Querying galactic extinction for %d objects...", len(merged_df))
                merged_df = get_extinction_batch(merged_df, cache=self.cache)
                n_with_ext = merged_df['A_g'].notna().sum()
                logger.info("Extinction values obtained for %d/%d objects",
                            n_with_ext, len(merged_df))
            except Exception as e:
                logger.warning("Extinction lookup failed: %s", e)

        if len(merged_df) > 0:
            try:
                self.cache.cache_merged_alerts(merged_df)
            except Exception as e:
                logger.debug(f"Could not cache merged alerts: {e}")

        return merged_df

    def _normalize_broker_probs(self, df: pd.DataFrame, broker: str) -> pd.DataFrame:
        """Ensure every broker DataFrame has a sn_ia_prob column.

        ALeRCE: already has sn_ia_prob from query_objects probability field.
        ANTARES: no real P(Ia) classifier. We compute a heuristic proxy
                 from ztf_sgscore1 (star/galaxy) and duration. Capped at 0.50
                 so that ANTARES alone cannot drive the pipeline filter;
                 only ALeRCE's ML probabilities should do that.
        """
        if 'sn_ia_prob' in df.columns:
            return df

        if broker == 'ANTARES':
            df['sn_ia_prob'] = df.apply(self._compute_antares_proxy_prob, axis=1)

        return df

    @staticmethod
    def _compute_antares_proxy_prob(row) -> float:
        """Compute a heuristic P(Ia) proxy for ANTARES objects.

        ANTARES has no real P(Ia). This proxy uses:
        - Star/galaxy score (sgscore < 0.5 = galaxy-associated = good)
        - Duration (true SNe have durations < ~200 days)

        Returns a value in [0.05, 0.50] — capped well below ALeRCE's
        ML probabilities to reflect that this is a heuristic.
        """
        prob = 0.25  # agnostic prior for a DDF transient candidate

        sgscore = row.get('ztf_sgscore1')
        if pd.notna(sgscore):
            if sgscore < 0.3:
                prob += 0.15
            elif sgscore < 0.5:
                prob += 0.05
            else:
                prob -= 0.15

        duration = row.get('duration_days')
        if pd.notna(duration):
            if duration < 100:
                prob += 0.05
            elif duration > 500:
                prob -= 0.10

        return max(0.05, min(0.50, prob))

    def _deduplicate_by_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate alerts by coordinate proximity or shared ZTF object ID."""
        if len(df) == 0:
            return df

        unique_alerts = []
        processed_indices = set()
        tol_deg = self.match_tolerance / 3600

        # Build ZTF ID lookup for secondary matching
        has_ztf_id = 'ztf_object_id' in df.columns

        for idx, alert in df.iterrows():
            if idx in processed_indices:
                continue

            ra = alert['ra']
            dec = alert['dec']

            # Match by coordinates
            matches_mask = (
                (np.abs(df['ra'] - ra) < tol_deg) &
                (np.abs(df['dec'] - dec) < tol_deg)
            )

            # Also match by shared ZTF object ID
            if has_ztf_id:
                ztf_id = alert.get('ztf_object_id')
                oid = alert.get('object_id')
                if pd.notna(ztf_id) and ztf_id:
                    matches_mask = matches_mask | (df['ztf_object_id'] == ztf_id)
                if pd.notna(oid) and oid and 'object_id' in df.columns:
                    matches_mask = matches_mask | (df['object_id'] == oid)

            matched_alerts = df[matches_mask & ~df.index.isin(processed_indices)].copy()
            # Also include this row
            if idx not in matched_alerts.index:
                matched_alerts = pd.concat([df.loc[[idx]], matched_alerts])

            processed_indices.update(matched_alerts.index)

            merged_alert = self._merge_duplicate_detections(matched_alerts)
            unique_alerts.append(merged_alert)

        merged_df = pd.DataFrame(unique_alerts)
        n_merged = len(df) - len(merged_df)
        if n_merged > 0:
            logger.info(f"Cross-matched {n_merged} duplicates across brokers")
        logger.info(f"Deduplicated to {len(merged_df)} unique objects")

        return merged_df

    def _merge_duplicate_detections(self, alert_group: pd.DataFrame) -> Dict:
        """Merge duplicate detections of the same object from multiple brokers."""
        first_alert = alert_group.iloc[0].to_dict()

        unique_id = f"{first_alert['ra']:.5f}_{first_alert['dec']:.5f}"

        brokers = alert_group['broker_source'].unique().tolist()
        brokers_str = ','.join(sorted(brokers))

        merged = {
            'unique_id': unique_id,
            'ra': first_alert.get('ra'),
            'dec': first_alert.get('dec'),
            'discovery_date': first_alert.get('discovery_date'),
            'brokers_detected': brokers_str,
            'num_brokers': len(brokers),
            'broker_names': brokers,
        }

        # Add broker-specific data
        for broker in brokers:
            broker_alerts = alert_group[alert_group['broker_source'] == broker]
            if len(broker_alerts) == 0:
                continue
            alert = broker_alerts.iloc[0]

            # Type Ia probability (now guaranteed to exist after normalization)
            if pd.notna(alert.get('sn_ia_prob')):
                merged[f'classification_{broker}_ia_prob'] = float(alert['sn_ia_prob'])

            # Other SN type probabilities (from ALeRCE enrichment)
            for key_suffix, col_names in [
                ('ii_prob', ['prob_snii', 'sn_ii_prob']),
                ('ibc_prob', ['prob_snibc', 'sn_ibc_prob']),
                ('slsn_prob', ['prob_slsn']),
            ]:
                for cn in col_names:
                    if cn in alert and pd.notna(alert.get(cn)):
                        merged[f'classification_{broker}_{key_suffix}'] = float(alert[cn])
                        break

            merged[f'object_id_{broker}'] = alert.get('object_id')
            merged[f'magnitude_{broker}'] = alert.get('magnitude') or alert.get('brightest_mag')

            # Carry Fink-specific classifier scores (preserved separately from mean_ia_prob)
            if broker == 'Fink':
                # sn_score is the primary Fink SN classifier score
                sn_score = alert.get('sn_score') or alert.get('sn_ia_prob')
                if pd.notna(sn_score):
                    merged['sn_score'] = float(sn_score)
                # early_ia_score is Fink's early SN Ia classifier
                early_score = alert.get('early_ia_score')
                if pd.notna(early_score):
                    merged['early_ia_score'] = float(early_score)
                # Preserve Fink's diaObjectId
                fink_did = alert.get('diaObjectId') or alert.get('object_id')
                if pd.notna(fink_did) and str(fink_did).strip():
                    merged['rubin_dia_object_id'] = str(fink_did).strip()

            # Carry ANTARES-specific metadata and quality fields
            if broker == 'ANTARES':
                merged[f'tags_{broker}'] = alert.get('tags', '')
                merged[f'source_tag_{broker}'] = alert.get('source_tag', '')
                merged[f'sgscore_{broker}'] = alert.get('ztf_sgscore1')
                merged[f'duration_{broker}'] = alert.get('duration_days')
                merged[f'rb_score_{broker}'] = alert.get('ztf_rb')
                merged['ddf_field'] = alert.get('ddf_field', '')
                rubin_id = alert.get('rubin_dia_object_id', '')
                if rubin_id:
                    merged['rubin_dia_object_id'] = rubin_id

            if broker in ('ALeRCE', 'ALeRCE-LSST'):
                merged['ddf_field'] = merged.get('ddf_field') or alert.get('ddf_field', '')

            if broker == 'ALeRCE-LSST':
                # The ALeRCE LSST oid is the Rubin DIA object ID
                lsst_oid = alert.get('object_id', '')
                if lsst_oid:
                    merged['rubin_dia_object_id'] = str(lsst_oid)

        # Ensure rubin_dia_object_id exists even if only ALeRCE detected it
        if 'rubin_dia_object_id' not in merged:
            merged['rubin_dia_object_id'] = ''

        return merged

    def _add_classification_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mean_ia_prob, std_ia_prob, agreement_score columns."""
        if len(df) == 0:
            return df

        ia_probs = []
        for _, row in df.iterrows():
            probs = []
            for col in df.columns:
                if '_ia_prob' in col and 'mean_' not in col and 'std_' not in col and 'min_' not in col and 'max_' not in col:
                    val = row.get(col)
                    if pd.notna(val):
                        probs.append(float(val))

            if probs:
                ia_probs.append({
                    'mean_ia_prob': np.mean(probs),
                    'std_ia_prob': np.std(probs) if len(probs) > 1 else 0.0,
                    'min_ia_prob': np.min(probs),
                    'max_ia_prob': np.max(probs),
                    'agreement_score': 1.0 if len(probs) == 1 else (
                        1.0 if np.std(probs) < 0.1 else 0.5 if np.std(probs) < 0.25 else 0.0
                    )
                })
            else:
                ia_probs.append({
                    'mean_ia_prob': np.nan,
                    'std_ia_prob': np.nan,
                    'min_ia_prob': np.nan,
                    'max_ia_prob': np.nan,
                    'agreement_score': 0.0
                })

        stats_df = pd.DataFrame(ia_probs)
        return pd.concat([df.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)

    def get_high_confidence_candidates(self, df: pd.DataFrame,
                                      min_ia_probability: float = 0.5,
                                      min_agreement: float = 0.0,
                                      num_brokers: int = 1) -> pd.DataFrame:
        """Filter for high-confidence Type Ia candidates."""
        filtered = df.copy()
        initial = len(filtered)

        if 'mean_ia_prob' in filtered.columns:
            filtered = filtered[filtered['mean_ia_prob'] >= min_ia_probability]
            logger.info(f"  After P(Ia) >= {min_ia_probability}: {len(filtered)} / {initial}")

        if min_agreement > 0 and 'agreement_score' in filtered.columns:
            before = len(filtered)
            filtered = filtered[filtered['agreement_score'] >= min_agreement]
            logger.info(f"  After agreement >= {min_agreement}: {len(filtered)} / {before}")

        if num_brokers > 1 and 'num_brokers' in filtered.columns:
            before = len(filtered)
            filtered = filtered[filtered['num_brokers'] >= num_brokers]
            logger.info(f"  After num_brokers >= {num_brokers}: {len(filtered)} / {before}")

        logger.info(f"Filtered to {len(filtered)} high-confidence Type Ia candidates")

        return filtered
