"""Fink broker client for current Rubin/LSST prompt photometry.

Fink processes the Rubin alert stream in real time and provides:
- DiaSource detections (alert-triggered)
- Forced photometry at DiaObject positions (dense multi-band)
- Classification scores (SN vs others, early SN Ia)
- Difference-image cutouts
- Cross-matches with SIMBAD, Gaia DR3, Legacy Survey

This client fetches up-to-date light curves for SN candidates
discovered by ALeRCE (or any broker), using Fink's LSST REST API.

API docs: https://api.lsst.fink-portal.org
"""

import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import requests

from .base_client import BaseBrokerClient

logger = logging.getLogger(__name__)

FINK_LSST_URL = "https://api.lsst.fink-portal.org"

# Fink column prefixes:
#   r: = raw Rubin alert fields
#   f: = Fink value-added fields
#   v: = virtual (computed) fields

# Photometry columns to request
_PHOT_COLS = ",".join([
    "r:diaObjectId", "r:diaSourceId",
    "r:midpointMjdTai", "r:band",
    "r:psfFlux", "r:psfFluxErr",
    "r:scienceFlux", "r:scienceFluxErr",
    "r:snr", "r:ra", "r:dec",
])

_FP_COLS = ",".join([
    "r:diaObjectId", "r:diaForcedSourceId",
    "r:midpointMjdTai", "r:band",
    "r:psfFlux", "r:psfFluxErr",
    "r:scienceFlux", "r:scienceFluxErr",
    "r:visit",
])


class FinkLSSTClient(BaseBrokerClient):
    """Client for Fink's Rubin/LSST alert API.

    Provides current prompt photometry for transient candidates,
    complementing ALeRCE's classification with Fink's real-time
    DiaSource and forced photometry data.

    Parameters
    ----------
    base_url : str
        Fink LSST API base URL.
    timeout : int
        HTTP request timeout in seconds.
    cache_dir : str
        Directory for caching data.
    """

    def __init__(self, base_url: str = FINK_LSST_URL, timeout: int = 60,
                 cache_dir: str = './cache/data'):
        super().__init__(broker_name='Fink', cache_dir=cache_dir)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        """Check if the Fink LSST API is reachable."""
        try:
            r = requests.get(f"{self.base_url}/api/v1/statistics",
                             timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core photometry queries
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: dict) -> Optional[list]:
        """POST to Fink API, return parsed JSON list or None."""
        url = f"{self.base_url}{endpoint}"
        payload["output-format"] = "json"
        try:
            r = requests.post(url, json=payload,
                              headers={"Content-Type": "application/json"},
                              timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
            logger.warning("Fink %s returned non-list: %s", endpoint,
                           type(data).__name__)
            return None
        except requests.exceptions.HTTPError as e:
            logger.warning("Fink %s HTTP error: %s", endpoint, e)
            return None
        except Exception as e:
            logger.warning("Fink %s error: %s", endpoint, e)
            return None

    def query_sources(self, dia_object_id: str) -> Optional[pd.DataFrame]:
        """Fetch DiaSource detections for one object.

        Returns DataFrame with mjd, band, flux (nJy), flux_err,
        science_flux, snr.
        """
        data = self._post("/api/v1/sources", {
            "diaObjectId": str(dia_object_id),
            "columns": _PHOT_COLS,
        })
        if not data:
            return None

        df = pd.DataFrame(data)
        df = self._normalize_columns(df)
        df["source"] = "detection"
        df["survey"] = "Rubin"
        return df

    def batch_query_source_counts(self, dia_object_ids: List[str],
                                    min_sources: int = 5
                                    ) -> Dict[str, int]:
        """Batch query source counts for multiple diaObjectIds.

        Efficiently pre-filters candidates before expensive light curve
        fetches. Uses comma-separated diaObjectIds in a single API call.

        Parameters
        ----------
        dia_object_ids : list of str
            List of diaObjectIds to query.
        min_sources : int
            Minimum source count threshold (for logging stats).

        Returns
        -------
        dict mapping diaObjectId (str) → source count (int).
        Objects not found return 0.
        """
        if not dia_object_ids:
            return {}

        # Deduplicate and convert to strings
        unique_ids = list(set(str(oid) for oid in dia_object_ids if oid))
        if not unique_ids:
            return {}

        # Batch query with minimal columns for efficiency
        batch_size = 100  # API may have limits; chunk if needed
        all_counts = {}

        for i in range(0, len(unique_ids), batch_size):
            chunk = unique_ids[i:i + batch_size]
            ids_str = ",".join(chunk)

            data = self._post("/api/v1/sources", {
                "diaObjectId": ids_str,
                "columns": "r:diaObjectId,r:diaSourceId",
            })

            if data:
                # Count sources per object
                chunk_counts = {}
                for row in data:
                    oid = str(row.get("r:diaObjectId", ""))
                    if oid:
                        chunk_counts[oid] = chunk_counts.get(oid, 0) + 1
                all_counts.update(chunk_counts)

            # Mark missing objects as 0
            for oid in chunk:
                if oid not in all_counts:
                    all_counts[oid] = 0

        # Log summary
        n_total = len(all_counts)
        n_above = sum(1 for c in all_counts.values() if c >= min_sources)
        logger.info("Fink batch source counts: %d/%d objects have >= %d sources",
                    n_above, n_total, min_sources)

        return all_counts

    def query_forced_photometry(self, dia_object_id: str
                                ) -> Optional[pd.DataFrame]:
        """Fetch forced photometry for one object.

        Returns dense multi-band measurements at every visit,
        including non-detections (flux ~ 0).
        """
        data = self._post("/api/v1/fp", {
            "diaObjectId": str(dia_object_id),
            "columns": _FP_COLS,
        })
        if not data:
            return None

        df = pd.DataFrame(data)
        df = self._normalize_columns(df)
        df["source"] = "forced_phot"
        df["survey"] = "Rubin"
        return df

    def prefilter_by_source_count(self, candidates_df: pd.DataFrame,
                                    min_sources: int = 5,
                                    id_column: str = None
                                    ) -> pd.DataFrame:
        """Pre-filter candidates by Fink source count.

        Efficiently removes candidates with too few detections before
        expensive light curve fetches.

        Parameters
        ----------
        candidates_df : DataFrame
            Candidates with diaObjectId column.
        min_sources : int
            Minimum number of DiaSource detections required.
        id_column : str, optional
            Column containing diaObjectId. Auto-detected if None.

        Returns
        -------
        Filtered DataFrame with only candidates meeting threshold.
        Adds 'fink_source_count' column.
        """
        # Find the diaObjectId column
        if id_column is None:
            for col in ['diaObjectId', 'fink_diaObjectId', 'object_id',
                        'r:diaObjectId']:
                if col in candidates_df.columns:
                    id_column = col
                    break

        if id_column is None or id_column not in candidates_df.columns:
            logger.warning("No diaObjectId column found for pre-filtering")
            return candidates_df

        # Get unique IDs
        dia_ids = candidates_df[id_column].dropna().astype(str).unique().tolist()
        if not dia_ids:
            logger.warning("No valid diaObjectIds to pre-filter")
            return candidates_df

        # Batch query counts
        counts = self.batch_query_source_counts(dia_ids, min_sources=min_sources)

        # Add count column and filter
        df = candidates_df.copy()
        df['fink_source_count'] = df[id_column].astype(str).map(
            lambda x: counts.get(x, 0)
        )

        n_before = len(df)
        df_filtered = df[df['fink_source_count'] >= min_sources].copy()
        n_after = len(df_filtered)

        logger.info("Pre-filter: %d -> %d candidates (>= %d sources)",
                    n_before, n_after, min_sources)

        return df_filtered

    def get_light_curve(self, dia_object_id: str,
                        include_forced: bool = True
                        ) -> Optional[pd.DataFrame]:
        """Get complete light curve: detections + forced photometry.

        Combines both source types, converts flux to magnitudes,
        and returns sorted by MJD.
        """
        frames = []

        det = self.query_sources(dia_object_id)
        if det is not None and len(det) > 0:
            frames.append(det)
            logger.info("  Fink detections: %d points (%s)",
                        len(det), sorted(det["band"].unique()))

        if include_forced:
            fp = self.query_forced_photometry(dia_object_id)
            if fp is not None and len(fp) > 0:
                frames.append(fp)
                logger.info("  Fink forced phot: %d points (%s)",
                            len(fp), sorted(fp["band"].unique()))

        if not frames:
            logger.warning("No Fink photometry for diaObjectId=%s",
                           dia_object_id)
            return None

        combined = pd.concat(frames, ignore_index=True)
        combined = self._flux_to_mag(combined)
        combined = combined.sort_values("mjd").reset_index(drop=True)

        logger.info("Fink light curve for %s: %d points, %s, "
                     "MJD %.1f-%.1f",
                     dia_object_id, len(combined),
                     sorted(combined["band"].unique()),
                     combined["mjd"].min(), combined["mjd"].max())
        return combined

    # ------------------------------------------------------------------
    # Candidate discovery (cone search, tags)
    # ------------------------------------------------------------------

    def cone_search(self, ra: float, dec: float, radius_arcsec: float = 5.0,
                    n: int = 1000) -> Optional[pd.DataFrame]:
        """Search for Fink/LSST objects by position.

        Parameters
        ----------
        ra, dec : float
            Center position (J2000, degrees).
        radius_arcsec : float
            Search radius in arcseconds.
        n : int
            Maximum number of results.

        Returns
        -------
        DataFrame with Fink alert data (one row per DiaSource).
        """
        data = self._post("/api/v1/conesearch", {
            "ra": ra, "dec": dec,
            "radius": radius_arcsec,
            "n": n,
        })
        if not data:
            return None
        return pd.DataFrame(data)

    def query_sn_candidates(self, tag: str = "sn_near_galaxy_candidate",
                            n: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch SN candidates by Fink classification tag.

        Available tags:
        - 'sn_near_galaxy_candidate': SN candidates near known galaxies
        - 'extragalactic_new_candidate': new extragalactic transients
        - 'extragalactic_lt20mag_candidate': bright extragalactic
        - 'hostless_candidate': transients without obvious host
        - 'in_tns': objects reported to TNS

        Returns
        -------
        DataFrame with one row per alert, including classification
        scores and photometry.
        """
        data = self._post("/api/v1/tags", {"tag": tag, "n": n})
        if not data:
            return None
        df = pd.DataFrame(data)
        logger.info("Fink tag '%s': %d results", tag, len(df))
        return df

    # ------------------------------------------------------------------
    # Cross-match: ALeRCE candidates → Fink diaObjectIds
    # ------------------------------------------------------------------

    def crossmatch_candidates(self, candidates_df: pd.DataFrame,
                              radius_arcsec: float = 2.0
                              ) -> pd.DataFrame:
        """Cross-match broker candidates to Fink diaObjectIds.

        For each candidate (ra, dec), searches Fink and finds the
        nearest matching diaObjectId.
        """
        fink_ids = []
        fink_seps = []
        fink_sn_scores = []

        for _, row in candidates_df.iterrows():
            try:
                result = self.cone_search(
                    float(row["ra"]), float(row["dec"]),
                    radius_arcsec=radius_arcsec, n=5
                )
                if result is not None and len(result) > 0:
                    # Take the nearest match
                    if "v:separation_degree" in result.columns:
                        result = result.sort_values("v:separation_degree")
                    best = result.iloc[0]
                    fink_ids.append(str(best.get("r:diaObjectId", "")))
                    sep = best.get("v:separation_degree", np.nan)
                    fink_seps.append(float(sep) * 3600 if pd.notna(sep) else np.nan)
                    fink_sn_scores.append(
                        best.get("f:clf_snnSnVsOthers_score", np.nan)
                    )
                else:
                    fink_ids.append(None)
                    fink_seps.append(np.nan)
                    fink_sn_scores.append(np.nan)
            except Exception as e:
                logger.warning("Fink crossmatch failed for (%.5f, %.5f): %s",
                               row["ra"], row["dec"], e)
                fink_ids.append(None)
                fink_seps.append(np.nan)
                fink_sn_scores.append(np.nan)

        out = candidates_df.copy()
        out["fink_diaObjectId"] = fink_ids
        out["fink_sep_arcsec"] = fink_seps
        out["fink_sn_score"] = fink_sn_scores

        n_matched = out["fink_diaObjectId"].notna().sum()
        logger.info("Fink cross-match: %d / %d candidates matched",
                    n_matched, len(out))
        return out

    def get_photometry_for_candidates(self, candidates_df: pd.DataFrame,
                                      include_forced: bool = True,
                                      match_radius_arcsec: float = 2.0
                                      ) -> Dict[str, Optional[pd.DataFrame]]:
        """Cross-match candidates and fetch all Fink light curves.

        This is the main pipeline integration point:
        1. Cross-match each candidate position to Fink diaObjectIds
        2. Fetch complete light curves from Fink

        Parameters
        ----------
        candidates_df : DataFrame
            Must have 'ra', 'dec', and 'object_id' or 'unique_id'.
        include_forced : bool
            Include forced photometry.
        match_radius_arcsec : float
            Cross-match radius.

        Returns
        -------
        dict mapping candidate object_id → light curve DataFrame.
        """
        id_col = "object_id"
        if id_col not in candidates_df.columns:
            id_col = "unique_id"

        matched = self.crossmatch_candidates(
            candidates_df, radius_arcsec=match_radius_arcsec
        )

        light_curves = {}
        for _, row in matched.iterrows():
            oid = row[id_col]
            fink_id = row.get("fink_diaObjectId")

            if pd.isna(fink_id) or fink_id is None:
                light_curves[oid] = None
                continue

            try:
                lc = self.get_light_curve(str(fink_id),
                                          include_forced=include_forced)
                light_curves[oid] = lc
            except Exception as e:
                logger.warning("Failed to get Fink LC for %s "
                               "(fink_id=%s): %s", oid, fink_id, e)
                light_curves[oid] = None

        n_lc = sum(1 for v in light_curves.values() if v is not None)
        logger.info("Fink photometry: %d / %d candidates have light curves",
                    n_lc, len(candidates_df))
        return light_curves

    def get_classifications(self, candidates_df: pd.DataFrame,
                             radius_arcsec: float = 2.0
                             ) -> pd.DataFrame:
        """Get Fink SN classifications for candidates by coordinate cross-match.

        Useful for enriching ANTARES-only candidates that lack ML classifiers.
        Uses cone search to find Fink objects and retrieves their SN classifier
        scores (f:clf_snnSnVsOthers_score).

        Parameters
        ----------
        candidates_df : DataFrame
            Must have 'ra', 'dec' columns. Optionally 'diaObjectId' or 'object_id'.
        radius_arcsec : float
            Cross-match radius in arcseconds.

        Returns
        -------
        DataFrame with original columns plus:
            - fink_sn_score: Fink SN vs Others classifier score (0-1)
            - fink_diaObjectId: Matched Fink diaObjectId
            - fink_sep_arcsec: Separation to matched object
        """
        fink_scores = []
        fink_ids = []
        fink_seps = []

        for idx, row in candidates_df.iterrows():
            ra, dec = row.get('ra'), row.get('dec')
            if pd.isna(ra) or pd.isna(dec):
                fink_scores.append(np.nan)
                fink_ids.append(None)
                fink_seps.append(np.nan)
                continue

            try:
                result = self.cone_search(float(ra), float(dec),
                                          radius_arcsec=radius_arcsec, n=5)
                if result is not None and len(result) > 0:
                    # Sort by separation if available
                    if "v:separation_degree" in result.columns:
                        result = result.sort_values("v:separation_degree")
                    best = result.iloc[0]

                    fink_ids.append(str(best.get("r:diaObjectId", "")))
                    sep = best.get("v:separation_degree", np.nan)
                    fink_seps.append(float(sep) * 3600 if pd.notna(sep) else np.nan)

                    # Get SN classifier score
                    sn_score = best.get("f:clf_snnSnVsOthers_score", np.nan)
                    fink_scores.append(float(sn_score) if pd.notna(sn_score) else np.nan)
                else:
                    fink_scores.append(np.nan)
                    fink_ids.append(None)
                    fink_seps.append(np.nan)

            except Exception as e:
                logger.debug("Fink classification lookup failed for (%.4f, %.4f): %s",
                             ra, dec, e)
                fink_scores.append(np.nan)
                fink_ids.append(None)
                fink_seps.append(np.nan)

        out = candidates_df.copy()
        out["fink_sn_score"] = fink_scores
        out["fink_diaObjectId"] = fink_ids
        out["fink_sep_arcsec"] = fink_seps

        n_matched = pd.notna(out["fink_sn_score"]).sum()
        logger.info("Fink classifications: %d / %d candidates have SN scores",
                    n_matched, len(out))
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename Fink r: prefixed columns to standard names."""
        rename = {
            "r:midpointMjdTai": "mjd",
            "r:band": "band",
            "r:psfFlux": "flux",
            "r:psfFluxErr": "flux_err",
            "r:scienceFlux": "science_flux",
            "r:scienceFluxErr": "science_flux_err",
            "r:snr": "snr",
            "r:ra": "ra",
            "r:dec": "dec",
            "r:diaObjectId": "diaObjectId",
            "r:diaSourceId": "diaSourceId",
            "r:diaForcedSourceId": "diaForcedSourceId",
            "r:visit": "visit",
        }
        out = df.rename(columns={k: v for k, v in rename.items()
                                 if k in df.columns})
        return out

    @staticmethod
    def _flux_to_mag(df: pd.DataFrame) -> pd.DataFrame:
        """Convert psfFlux (nJy) to AB magnitude.

        AB mag = -2.5 * log10(flux_nJy) + 31.4
        Negative/zero flux → magnitude = NaN.
        """
        flux = df["flux"].values.astype(float)
        flux_err = df["flux_err"].values.astype(float)

        valid = flux > 0
        magnitude = np.full(len(df), np.nan)
        mag_err = np.full(len(df), np.nan)

        magnitude[valid] = -2.5 * np.log10(flux[valid]) + 31.4
        mag_err[valid] = (2.5 / np.log(10)) * (flux_err[valid] / flux[valid])

        df = df.copy()
        df["magnitude"] = magnitude
        df["mag_err"] = mag_err
        return df

    # ------------------------------------------------------------------
    # BaseBrokerClient abstract method implementations
    # ------------------------------------------------------------------

    def query_alerts(self,
                     class_name: str = 'SN Ia',
                     min_probability: float = 0.7,
                     days_back: int = 30,
                     **kwargs) -> pd.DataFrame:
        """Query Fink for SN candidates using the tag-based search.

        Maps to query_sn_candidates() with appropriate tag.
        """
        # Map class names to Fink tags
        tag_map = {
            'SN Ia': 'sn_near_galaxy_candidate',
            'SNIa': 'sn_near_galaxy_candidate',
            'SN': 'extragalactic_new_candidate',
        }
        tag = tag_map.get(class_name, 'sn_near_galaxy_candidate')

        result = self.query_sn_candidates(tag=tag, n=kwargs.get('limit', 1000))
        if result is None:
            return pd.DataFrame()

        # Filter by SN score if available
        if 'f:clf_snnSnVsOthers_score' in result.columns:
            result = result[result['f:clf_snnSnVsOthers_score'] >= min_probability]

        return result

    def get_stamps(self, object_id: str, ra: float, dec: float) -> Dict[str, Any]:
        """Retrieve postage stamp images (not implemented for Fink).

        Fink provides stamps via a different endpoint; return empty for now.
        """
        logger.debug("Stamp retrieval not implemented for Fink")
        return {}
