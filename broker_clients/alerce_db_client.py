"""Direct PostgreSQL client for ALeRCE's ZTF database.

Provides bulk query access to ALeRCE's read-only PostgreSQL database,
which is much faster than the REST API for large result sets.

Credentials are fetched from the ALeRCE usecases repository.
Requires: psycopg2-binary, sqlalchemy

Reference: https://github.com/alercebroker/usecases
"""

import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CREDENTIALS_URL = (
    "https://raw.githubusercontent.com/alercebroker/usecases/"
    "master/alercereaduser_v4.json"
)

# ALeRCE light curve classifier class names for supernovae
SN_CLASSES = ('SNIa', 'SNIbc', 'SNII', 'SLSN')

# All light curve classifier classes (for pivoting)
LC_CLASSIFIER_CLASSES = [
    'SNIa', 'SNIbc', 'SNII', 'SLSN',
    'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO',
    'DSCT', 'CEP', 'LPV', 'RRL', 'E', 'Periodic-Other',
]

# ZTF filter ID mapping
FID_MAP = {1: 'g', 2: 'r', 3: 'i'}


class AlerceDBClient:
    """Direct PostgreSQL client for ALeRCE's ZTF database."""

    def __init__(self):
        self._engine = None
        self._available = None

    @property
    def available(self) -> bool:
        """Check if psycopg2 is installed and credentials are reachable."""
        if self._available is None:
            try:
                import psycopg2  # noqa: F401
                import sqlalchemy  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.info("ALeRCE DB client unavailable: psycopg2 not installed")
        return self._available

    def connect(self):
        """Establish database connection using remote credentials."""
        if self._engine is not None:
            return

        import requests
        import sqlalchemy as sa

        resp = requests.get(CREDENTIALS_URL, timeout=10)
        resp.raise_for_status()
        params = resp.json()['params']

        url = (
            f"postgresql+psycopg2://{params['user']}:{params['password']}"
            f"@{params['host']}/{params['dbname']}"
        )
        self._engine = sa.create_engine(url, pool_pre_ping=True)

        # Verify connection
        with self._engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        logger.info("Connected to ALeRCE database at %s", params['host'])

    def _ensure_connected(self):
        if self._engine is None:
            self.connect()

    def _read_sql(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a read query and return a DataFrame."""
        self._ensure_connected()
        import sqlalchemy as sa
        with self._engine.connect() as conn:
            return pd.read_sql_query(sa.text(query), conn, params=params)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def query_sn_candidates(self, min_prob: float = 0.5,
                            classifier: str = 'lc_classifier',
                            max_rows: int = 50000) -> pd.DataFrame:
        """Query objects classified as supernovae by the light curve classifier.

        Parameters
        ----------
        min_prob : float
            Minimum probability threshold for SN classes.
        classifier : str
            Classifier name in the probability table.
        max_rows : int
            Safety limit on returned rows.

        Returns
        -------
        DataFrame with columns: oid, meanra, meandec, ndet, firstmjd,
            deltajd, g_r_max, classifier_name, class_name, probability, ranking
        """
        sn_classes_str = ", ".join(f"'{c}'" for c in SN_CLASSES)

        query = f"""
            SELECT
                object.oid, object.meanra, object.meandec, object.ndet,
                object.firstmjd, object.deltajd, object.g_r_max,
                probability.classifier_name, probability.class_name,
                probability.classifier_version,
                probability.ranking, probability.probability
            FROM
                object INNER JOIN probability
                ON object.oid = probability.oid
            WHERE
                probability.classifier_name = :classifier
                AND object.oid IN (
                    SELECT oid FROM probability
                    WHERE classifier_name = :classifier
                    AND class_name IN ({sn_classes_str})
                    AND ranking = 1
                    AND probability > :min_prob
                )
            LIMIT :max_rows
        """

        try:
            df = self._read_sql(query, {
                'classifier': classifier,
                'min_prob': min_prob,
                'max_rows': max_rows,
            })
            logger.info("ALeRCE DB: queried %d rows for SN candidates (prob > %.2f)",
                        len(df), min_prob)
            return df
        except Exception as e:
            logger.warning("ALeRCE DB query_sn_candidates failed: %s", e)
            return pd.DataFrame()

    def query_probabilities(self, oids: List[str],
                            classifier: str = 'lc_classifier') -> pd.DataFrame:
        """Get all class probabilities for a list of objects.

        Returns pivoted DataFrame: one row per OID, columns for each class probability.
        """
        if not oids:
            return pd.DataFrame()

        oids_str = ", ".join(f"'{o}'" for o in oids)

        query = f"""
            SELECT oid, class_name, probability, ranking
            FROM probability
            WHERE classifier_name = :classifier
            AND oid IN ({oids_str})
        """

        try:
            df = self._read_sql(query, {'classifier': classifier})
            if len(df) == 0:
                return pd.DataFrame()

            # Pivot to one row per OID with columns like prob_SNIa, prob_SNII, etc.
            pivoted = df.pivot_table(
                index='oid', columns='class_name',
                values='probability', aggfunc='first',
            )
            pivoted.columns = [f'prob_{c}' for c in pivoted.columns]
            pivoted = pivoted.reset_index()

            # Add ranking=1 class
            top_class = df[df['ranking'] == 1].set_index('oid')['class_name']
            pivoted = pivoted.merge(
                top_class.rename('top_class'), left_on='oid', right_index=True, how='left'
            )

            logger.info("ALeRCE DB: probabilities for %d objects", len(pivoted))
            return pivoted

        except Exception as e:
            logger.warning("ALeRCE DB query_probabilities failed: %s", e)
            return pd.DataFrame()

    def query_features(self, oids: List[str], prefix: str = 'SPM') -> pd.DataFrame:
        """Get SPM (or other) features for a list of objects.

        Returns pivoted DataFrame with columns like SPM_A_1, SPM_A_2
        (name_fid format, where fid 1=g, 2=r).
        """
        if not oids:
            return pd.DataFrame()

        oids_str = ", ".join(f"'{o}'" for o in oids)

        query = f"""
            SELECT oid, name, value, fid
            FROM feature
            WHERE LEFT(name, :prefix_len) = :prefix
            AND oid IN ({oids_str})
        """

        try:
            df = self._read_sql(query, {'prefix': prefix, 'prefix_len': len(prefix)})
            if len(df) == 0:
                return pd.DataFrame()

            # Create name_fid key and pivot
            df['name_fid'] = df['name'] + '_' + df['fid'].astype(str)
            pivoted = df.pivot_table(
                index='oid', columns='name_fid',
                values='value', aggfunc='first',
            )
            pivoted = pivoted.reset_index()

            logger.info("ALeRCE DB: SPM features for %d objects (%d columns)",
                        len(pivoted), len(pivoted.columns) - 1)
            return pivoted

        except Exception as e:
            logger.warning("ALeRCE DB query_features failed: %s", e)
            return pd.DataFrame()

    def query_detections(self, oids: List[str]) -> pd.DataFrame:
        """Get light curve detections for a list of objects.

        Returns DataFrame with columns: oid, mjd, fid, band, magpsf, sigmapsf.
        """
        if not oids:
            return pd.DataFrame()

        oids_str = ", ".join(f"'{o}'" for o in oids)

        query = f"""
            SELECT oid, mjd, fid, magpsf, sigmapsf
            FROM detection
            WHERE oid IN ({oids_str})
            ORDER BY oid, mjd
        """

        try:
            df = self._read_sql(query)
            if len(df) > 0:
                df['band'] = df['fid'].map(FID_MAP)
            logger.info("ALeRCE DB: %d detections for %d objects",
                        len(df), df['oid'].nunique() if len(df) > 0 else 0)
            return df

        except Exception as e:
            logger.warning("ALeRCE DB query_detections failed: %s", e)
            return pd.DataFrame()

    def query_ps1_host(self, oids: List[str]) -> pd.DataFrame:
        """Get PanSTARRS host galaxy data for a list of objects.

        Returns DataFrame with columns: oid, sgmag1, srmag1, simag1, szmag1,
            sgscore1, g_r_host, r_i_host.
        """
        if not oids:
            return pd.DataFrame()

        oids_str = ", ".join(f"'{o}'" for o in oids)

        query = f"""
            SELECT oid, sgmag1, srmag1, simag1, szmag1, sgscore1
            FROM ps1_ztf
            WHERE oid IN ({oids_str})
        """

        try:
            df = self._read_sql(query)
            if len(df) > 0:
                df['g_r_host'] = df['sgmag1'] - df['srmag1']
                df['r_i_host'] = df['srmag1'] - df['simag1']
            logger.info("ALeRCE DB: PS1 host data for %d objects", len(df))
            return df

        except Exception as e:
            logger.warning("ALeRCE DB query_ps1_host failed: %s", e)
            return pd.DataFrame()

    def query_magstats(self, oids: List[str]) -> pd.DataFrame:
        """Get magnitude statistics per band for a list of objects."""
        if not oids:
            return pd.DataFrame()

        oids_str = ", ".join(f"'{o}'" for o in oids)

        query = f"""
            SELECT oid, fid, ndet, magmin, magmax, magmean,
                   dmdt_first, dm_first, dt_first
            FROM magstat
            WHERE oid IN ({oids_str})
        """

        try:
            df = self._read_sql(query)
            if len(df) > 0:
                df['band'] = df['fid'].map(FID_MAP)
            logger.info("ALeRCE DB: magstats for %d objects", len(df))
            return df

        except Exception as e:
            logger.warning("ALeRCE DB query_magstats failed: %s", e)
            return pd.DataFrame()

    def crossmatch_positions(self, positions: List[tuple],
                             radius_arcsec: float = 2.0,
                             min_dec: float = -32.0) -> Dict[Any, str]:
        """Batch cross-match positions against ZTF object table.

        Uses box query for speed, then filters by exact angular distance.

        Parameters
        ----------
        positions : list of (id, ra, dec) tuples
            Positions to cross-match. id can be any hashable type.
        radius_arcsec : float
            Cross-match radius in arcseconds.
        min_dec : float
            Skip positions below this declination (no ZTF coverage).

        Returns
        -------
        dict of id -> ztf_oid for matched positions
        """
        if not positions:
            return {}

        # Filter to positions with ZTF coverage (dec > -32)
        valid_positions = [(pid, ra, dec) for pid, ra, dec in positions
                          if dec > min_dec]
        if not valid_positions:
            logger.info("ALeRCE DB: no positions with ZTF coverage (all dec < %.0f)",
                       min_dec)
            return {}

        radius_deg = radius_arcsec / 3600.0
        results = {}

        # Build UNION query for all positions (more efficient than N queries)
        # Use box query: |ra - ra0| < r/cos(dec) AND |dec - dec0| < r
        union_parts = []
        for pid, ra, dec in valid_positions:
            # RA tolerance depends on declination
            cos_dec = np.cos(np.radians(dec))
            ra_tol = radius_deg / max(cos_dec, 0.1)  # avoid div by zero near poles
            dec_tol = radius_deg

            # Subquery for this position
            part = f"""
                SELECT '{pid}' as query_id, oid, meanra, meandec,
                    SQRT(POWER((meanra - {ra}) * {cos_dec}, 2) +
                         POWER(meandec - {dec}, 2)) * 3600 as sep_arcsec
                FROM object
                WHERE meanra BETWEEN {ra - ra_tol} AND {ra + ra_tol}
                  AND meandec BETWEEN {dec - dec_tol} AND {dec + dec_tol}
            """
            union_parts.append(part)

        # Combine with UNION ALL (faster than UNION for large result sets)
        # Limit results per position to avoid runaway queries
        query = " UNION ALL ".join(union_parts)
        query = f"""
            SELECT * FROM ({query}) AS matches
            WHERE sep_arcsec < {radius_arcsec}
            ORDER BY query_id, sep_arcsec
        """

        try:
            df = self._read_sql(query)
            if len(df) == 0:
                logger.info("ALeRCE DB: no ZTF cross-matches for %d positions",
                           len(valid_positions))
                return {}

            # Keep only closest match per query_id
            df_closest = df.drop_duplicates('query_id', keep='first')
            results = dict(zip(df_closest['query_id'], df_closest['oid']))

            logger.info("ALeRCE DB: cross-matched %d/%d positions to ZTF objects",
                       len(results), len(valid_positions))
            return results

        except Exception as e:
            logger.warning("ALeRCE DB crossmatch_positions failed: %s", e)
            return {}
