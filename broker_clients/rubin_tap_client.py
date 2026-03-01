"""Rubin Science Platform TAP client for authoritative LSST photometry.

Uses pyvo to query the RSP TAP service for DiaObject, DiaSource, and
ForcedSourceOnDiaObject tables.  Designed to work alongside ALeRCE
(which provides discovery and classification) by cross-matching ALeRCE
candidates to RSP DiaObjects and returning complete forced + unforced
light curves.

Authentication: set the RSP_TOKEN environment variable with a bearer
token generated at https://data.lsst.cloud/auth/tokens
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# RSP TAP endpoint
DEFAULT_TAP_URL = "https://data.lsst.cloud/api/tap"

# Default schema for Rubin data products (DP1, adjust for future DRs)
DEFAULT_SCHEMA = "dp1"

# LSST band names
LSST_BANDS = ['u', 'g', 'r', 'i', 'z', 'y']

# Cross-match radius for ALeRCE → RSP matching (arcsec)
DEFAULT_MATCH_RADIUS_ARCSEC = 1.5


class RubinTAPClient:
    """Client for querying Rubin Science Platform via TAP.

    Fetches authoritative LSST photometry (difference-image detections
    and forced photometry) for supernova candidates discovered by alert
    brokers.

    Parameters
    ----------
    token : str, optional
        RSP bearer token.  If not provided, reads from the RSP_TOKEN
        environment variable or ~/.rsp_token file.
    tap_url : str
        TAP service URL.
    schema : str
        Database schema name (e.g. 'dp1', 'dp02_dc2_catalogs').
    """

    def __init__(self, token: Optional[str] = None,
                 tap_url: str = DEFAULT_TAP_URL,
                 schema: str = DEFAULT_SCHEMA):
        self.tap_url = tap_url
        self.schema = schema
        self._token = token or self._resolve_token()
        self._service = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_token() -> Optional[str]:
        """Resolve RSP token from environment or file."""
        # 1. Environment variable
        token = os.environ.get("RSP_TOKEN")
        if token:
            logger.debug("RSP token loaded from RSP_TOKEN env var")
            return token.strip()

        # 2. Token file
        token_file = Path.home() / ".rsp_token"
        if token_file.exists():
            token = token_file.read_text().strip()
            if token:
                logger.debug("RSP token loaded from %s", token_file)
                return token

        return None

    @property
    def available(self) -> bool:
        """Check whether pyvo is installed and a token is configured."""
        if not self._token:
            return False
        try:
            import pyvo  # noqa: F401
            return True
        except ImportError:
            return False

    def connect(self):
        """Create authenticated TAP service connection.

        Raises RuntimeError if token is missing or pyvo is not installed.
        """
        if self._service is not None:
            return  # already connected

        if not self._token:
            raise RuntimeError(
                "No RSP token found.  Set the RSP_TOKEN environment variable "
                "or create ~/.rsp_token with your bearer token."
            )

        try:
            import pyvo
            from pyvo.auth import AuthSession, CredentialStore
        except ImportError:
            raise RuntimeError(
                "pyvo is required for RSP TAP access.  "
                "Install with: pip install pyvo"
            )

        cred = CredentialStore()
        cred.set_password("x-oauth2-bearer", self._token)
        auth = AuthSession()
        auth.credentials = cred

        self._service = pyvo.dal.TAPService(self.tap_url, session=auth)
        logger.info("Connected to RSP TAP service at %s", self.tap_url)

    def _ensure_connected(self):
        """Lazy-connect on first query."""
        if self._service is None:
            self.connect()

    def check_data_coverage(self) -> dict:
        """Report what data is available: date range, fields, staleness.

        Returns dict with min_mjd, max_mjd, min_date, max_date,
        staleness_days, n_dia_objects, n_dia_sources, field_coverage.
        """
        self._ensure_connected()
        from astropy.time import Time

        info = {}

        # DiaSource date range
        result = self._run_query(f"""
            SELECT MIN(midpointMjdTai) AS min_mjd,
                   MAX(midpointMjdTai) AS max_mjd,
                   COUNT(*) AS n_sources
            FROM {self._table('DiaSource')}
        """)
        row = result.iloc[0]
        info['min_mjd'] = float(row['min_mjd'])
        info['max_mjd'] = float(row['max_mjd'])
        info['min_date'] = Time(info['min_mjd'], format='mjd').iso[:10]
        info['max_date'] = Time(info['max_mjd'], format='mjd').iso[:10]
        info['staleness_days'] = round(Time.now().mjd - info['max_mjd'])
        info['n_dia_sources'] = int(row['n_sources'])

        # DiaObject count
        result2 = self._run_query(f"""
            SELECT COUNT(*) AS cnt FROM {self._table('DiaObject')}
        """)
        info['n_dia_objects'] = int(result2.iloc[0]['cnt'])

        info['schema'] = self.schema
        return info

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _table(self, name: str) -> str:
        """Return fully qualified table name."""
        return f"{self.schema}.{name}"

    def _run_query(self, adql: str) -> pd.DataFrame:
        """Execute an ADQL query and return a DataFrame."""
        self._ensure_connected()
        logger.debug("ADQL query:\n%s", adql)
        result = self._service.search(adql)
        df = result.to_table().to_pandas()
        logger.debug("Query returned %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # DiaObject queries
    # ------------------------------------------------------------------

    def query_dia_objects(self, ra: float, dec: float,
                         radius_deg: float = 1.75) -> pd.DataFrame:
        """Find DiaObjects within a cone search.

        Parameters
        ----------
        ra, dec : float
            Center of search cone (J2000, degrees).
        radius_deg : float
            Search radius in degrees.

        Returns
        -------
        DataFrame with diaObjectId, ra, dec, nDiaSources, and
        available summary columns.
        """
        adql = f"""
            SELECT diaObjectId, ra, dec, nDiaSources
            FROM {self._table('DiaObject')}
            WHERE CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
            ) = 1
        """
        return self._run_query(adql)

    def crossmatch_to_dia_objects(self, candidates_df: pd.DataFrame,
                                  radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC
                                  ) -> pd.DataFrame:
        """Cross-match a candidates DataFrame to RSP DiaObjects.

        For each candidate row (must have 'ra' and 'dec' columns),
        finds the nearest DiaObject within radius_arcsec and adds
        'diaObjectId' and 'rsp_sep_arcsec' columns.

        Uses an ADQL point-in-circle query per candidate.  For large
        candidate lists, a bulk upload cross-match could be faster,
        but per-object queries are simpler and sufficient for DDF
        candidate lists (typically tens to low hundreds of objects).
        """
        self._ensure_connected()
        radius_deg = radius_arcsec / 3600.0

        dia_ids = []
        separations = []

        for _, row in candidates_df.iterrows():
            cra, cdec = float(row['ra']), float(row['dec'])
            adql = f"""
                SELECT diaObjectId, ra, dec,
                       DISTANCE(
                           POINT('ICRS', ra, dec),
                           POINT('ICRS', {cra}, {cdec})
                       ) AS sep_deg
                FROM {self._table('DiaObject')}
                WHERE CONTAINS(
                    POINT('ICRS', ra, dec),
                    CIRCLE('ICRS', {cra}, {cdec}, {radius_deg})
                ) = 1
                ORDER BY sep_deg ASC
            """
            try:
                matches = self._run_query(adql)
                if len(matches) > 0:
                    best = matches.iloc[0]
                    dia_ids.append(int(best['diaObjectId']))
                    separations.append(float(best['sep_deg']) * 3600.0)
                else:
                    dia_ids.append(None)
                    separations.append(np.nan)
            except Exception as e:
                logger.warning("RSP crossmatch failed for (%.5f, %.5f): %s",
                               cra, cdec, e)
                dia_ids.append(None)
                separations.append(np.nan)

        result = candidates_df.copy()
        result['diaObjectId'] = dia_ids
        result['rsp_sep_arcsec'] = separations

        n_matched = result['diaObjectId'].notna().sum()
        logger.info("RSP cross-match: %d / %d candidates matched "
                    "(radius=%.1f arcsec)", n_matched, len(result),
                    radius_arcsec)
        return result

    # ------------------------------------------------------------------
    # Light curve queries
    # ------------------------------------------------------------------

    def query_dia_sources(self, dia_object_id: int) -> pd.DataFrame:
        """Fetch alert-triggered detections (DiaSource) for one object.

        Returns DataFrame with mjd, band, flux (nJy), flux_err, snr,
        and science_flux (direct-image flux) when available.
        """
        adql = f"""
            SELECT diaSourceId, midpointMjdTai, band,
                   psfFlux, psfFluxErr,
                   scienceFlux, scienceFluxErr,
                   snr
            FROM {self._table('DiaSource')}
            WHERE diaObjectId = {dia_object_id}
            ORDER BY midpointMjdTai ASC
        """
        df = self._run_query(adql)
        if len(df) == 0:
            return df

        df = df.rename(columns={
            'midpointMjdTai': 'mjd',
            'psfFlux': 'flux',
            'psfFluxErr': 'flux_err',
            'scienceFlux': 'science_flux',
            'scienceFluxErr': 'science_flux_err',
        })
        df['source'] = 'detection'
        df['survey'] = 'Rubin'
        return df

    def query_forced_photometry(self, dia_object_id: int) -> pd.DataFrame:
        """Fetch forced photometry (ForcedSourceOnDiaObject) for one object.

        This provides a measurement at the object position for EVERY
        visit of the field, giving dense temporal coverage including
        non-detections.

        Includes both difference-image flux (psfDiffFlux) and direct-
        image flux (psfFlux).  MJD is obtained by joining to dp1.Visit.
        """
        adql = f"""
            SELECT f.diaObjectId, f.band,
                   v.expMidptMJD AS mjd,
                   f.psfDiffFlux, f.psfDiffFluxErr,
                   f.psfFlux AS science_flux,
                   f.psfFluxErr AS science_flux_err
            FROM {self._table('ForcedSourceOnDiaObject')} AS f
            JOIN {self._table('Visit')} AS v
              ON f.visit = v.visit
            WHERE f.diaObjectId = {dia_object_id}
            ORDER BY v.expMidptMJD ASC
        """
        df = self._run_query(adql)
        if len(df) == 0:
            return df

        df = df.rename(columns={
            'psfDiffFlux': 'flux',
            'psfDiffFluxErr': 'flux_err',
        })
        df['source'] = 'forced_phot'
        df['survey'] = 'Rubin'
        return df

    def get_light_curve(self, dia_object_id: int,
                        include_forced: bool = True) -> Optional[pd.DataFrame]:
        """Get complete light curve for a DiaObject.

        Combines alert-triggered detections and forced photometry into
        a single DataFrame with flux-to-magnitude conversion.

        Parameters
        ----------
        dia_object_id : int
            RSP DiaObject identifier.
        include_forced : bool
            If True, include forced photometry (recommended).

        Returns
        -------
        DataFrame with columns: mjd, band, flux, flux_err, magnitude,
        mag_err, survey, source.  Sorted by mjd.
        """
        frames = []

        # Alert-triggered detections
        try:
            det = self.query_dia_sources(dia_object_id)
            if len(det) > 0:
                frames.append(det)
                logger.info("  RSP detections: %d points", len(det))
        except Exception as e:
            logger.warning("Failed to query DiaSource for %s: %s",
                           dia_object_id, e)

        # Forced photometry
        if include_forced:
            try:
                fp = self.query_forced_photometry(dia_object_id)
                if len(fp) > 0:
                    frames.append(fp)
                    logger.info("  RSP forced phot: %d points", len(fp))
            except Exception as e:
                logger.warning("Failed to query forced phot for %s: %s",
                               dia_object_id, e)

        if not frames:
            logger.warning("No RSP photometry for diaObjectId=%s",
                           dia_object_id)
            return None

        combined = pd.concat(frames, ignore_index=True)

        # Convert flux (nJy) to AB magnitudes
        combined = self._flux_to_mag(combined)

        # Sort by time
        combined = combined.sort_values('mjd').reset_index(drop=True)

        logger.info("RSP light curve for diaObjectId=%s: %d points, "
                     "bands=%s", dia_object_id, len(combined),
                     sorted(combined['band'].unique()))
        return combined

    @staticmethod
    def _flux_to_mag(df: pd.DataFrame) -> pd.DataFrame:
        """Convert psfFlux (nJy) to AB magnitude.

        AB mag = -2.5 * log10(flux_nJy) + 31.4
        Negative/zero flux → magnitude = NaN (but flux preserved).
        """
        flux = df['flux'].values.astype(float)
        flux_err = df['flux_err'].values.astype(float)

        valid = flux > 0
        magnitude = np.full(len(df), np.nan)
        mag_err = np.full(len(df), np.nan)

        magnitude[valid] = -2.5 * np.log10(flux[valid]) + 31.4
        mag_err[valid] = (2.5 / np.log(10)) * (flux_err[valid] / flux[valid])

        df = df.copy()
        df['magnitude'] = magnitude
        df['mag_err'] = mag_err
        return df

    # ------------------------------------------------------------------
    # Batch operations for pipeline integration
    # ------------------------------------------------------------------

    def get_photometry_for_candidates(self, candidates_df: pd.DataFrame,
                                      include_forced: bool = True,
                                      match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC
                                      ) -> Dict[str, Optional[pd.DataFrame]]:
        """Cross-match candidates to RSP and fetch all light curves.

        This is the main entry point for pipeline integration:
        1. Cross-match each candidate's (ra, dec) to RSP DiaObjects.
        2. For each matched object, fetch complete photometry.

        Parameters
        ----------
        candidates_df : DataFrame
            Must have 'ra', 'dec', and an identifier column
            ('object_id' or 'unique_id').
        include_forced : bool
            Include forced photometry (recommended).
        match_radius_arcsec : float
            Cross-match radius in arcsec.

        Returns
        -------
        dict mapping candidate object_id → light curve DataFrame
        (or None if no RSP match).
        """
        self._ensure_connected()

        # Determine ID column
        id_col = 'object_id'
        if id_col not in candidates_df.columns:
            id_col = 'unique_id'
        if id_col not in candidates_df.columns:
            raise ValueError("candidates_df must have 'object_id' or 'unique_id' column")

        # Cross-match to RSP
        matched = self.crossmatch_to_dia_objects(
            candidates_df, radius_arcsec=match_radius_arcsec
        )

        # Fetch light curves for matched objects
        light_curves = {}
        n_matched = matched['diaObjectId'].notna().sum()
        logger.info("Fetching RSP photometry for %d matched candidates...",
                     n_matched)

        for _, row in matched.iterrows():
            oid = row[id_col]
            dia_id = row.get('diaObjectId')

            if pd.isna(dia_id) or dia_id is None:
                light_curves[oid] = None
                continue

            try:
                lc = self.get_light_curve(int(dia_id),
                                          include_forced=include_forced)
                light_curves[oid] = lc
            except Exception as e:
                logger.warning("Failed to get RSP light curve for %s "
                               "(diaObjectId=%s): %s", oid, dia_id, e)
                light_curves[oid] = None

        n_lc = sum(1 for v in light_curves.values() if v is not None)
        logger.info("RSP photometry: %d / %d candidates have light curves",
                     n_lc, len(candidates_df))
        return light_curves

    def query_dia_objects_bulk(self, ra: float, dec: float,
                               radius_deg: float = 1.75,
                               min_sources: int = 3) -> pd.DataFrame:
        """Bulk query DiaObjects in a DDF field.

        Useful for pre-loading the DiaObject catalog for a field so
        that cross-matching can be done locally rather than with
        per-object TAP queries.

        Parameters
        ----------
        ra, dec : float
            Field center (J2000, degrees).
        radius_deg : float
            Search radius.
        min_sources : int
            Minimum number of DiaSource detections.

        Returns
        -------
        DataFrame of DiaObjects in the field.
        """
        adql = f"""
            SELECT diaObjectId, ra, dec, nDiaSources
            FROM {self._table('DiaObject')}
            WHERE CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
            ) = 1
              AND nDiaSources >= {min_sources}
        """
        return self._run_query(adql)

    def crossmatch_local(self, candidates_df: pd.DataFrame,
                         dia_objects_df: pd.DataFrame,
                         radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC
                         ) -> pd.DataFrame:
        """Cross-match candidates to a pre-loaded DiaObject catalog.

        Much faster than per-object TAP queries for large candidate
        lists.  Call query_dia_objects_bulk() first to get the
        dia_objects_df for each DDF field.

        Parameters
        ----------
        candidates_df : DataFrame
            Must have 'ra', 'dec' columns.
        dia_objects_df : DataFrame
            Must have 'diaObjectId', 'ra', 'dec' columns.
        radius_arcsec : float
            Match radius.

        Returns
        -------
        candidates_df with added 'diaObjectId' and 'rsp_sep_arcsec'.
        """
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        if len(dia_objects_df) == 0:
            result = candidates_df.copy()
            result['diaObjectId'] = None
            result['rsp_sep_arcsec'] = np.nan
            return result

        cat_coord = SkyCoord(
            ra=dia_objects_df['ra'].values * u.deg,
            dec=dia_objects_df['dec'].values * u.deg,
        )

        dia_ids = []
        separations = []

        for _, row in candidates_df.iterrows():
            cand_coord = SkyCoord(ra=row['ra'] * u.deg,
                                  dec=row['dec'] * u.deg)
            sep = cand_coord.separation(cat_coord).arcsec
            idx_min = np.argmin(sep)
            min_sep = sep[idx_min]

            if min_sep <= radius_arcsec:
                dia_ids.append(int(dia_objects_df.iloc[idx_min]['diaObjectId']))
                separations.append(min_sep)
            else:
                dia_ids.append(None)
                separations.append(np.nan)

        result = candidates_df.copy()
        result['diaObjectId'] = dia_ids
        result['rsp_sep_arcsec'] = separations

        n_matched = result['diaObjectId'].notna().sum()
        logger.info("Local cross-match: %d / %d candidates matched",
                    n_matched, len(result))
        return result
