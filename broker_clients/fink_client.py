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
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import requests

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


class FinkLSSTClient:
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
    """

    def __init__(self, base_url: str = FINK_LSST_URL, timeout: int = 60):
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
