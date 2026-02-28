"""Local caching system for alerts and galaxy information."""

import os
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlertCache:
    """Cache manager for broker alerts and galaxy information."""

    def __init__(self, cache_dir: str = './cache/data', db_name: str = 'alerts_cache.db'):
        """
        Initialize alert cache.

        Args:
            cache_dir: Directory for cache storage
            db_name: SQLite database filename
        """
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, db_name)
        os.makedirs(cache_dir, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self):
        """Initialize SQLite database schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY,
                    object_id TEXT NOT NULL,
                    broker TEXT NOT NULL,
                    ra REAL,
                    dec REAL,
                    discovery_date TEXT,
                    classification JSON,
                    photometry JSON,
                    metadata JSON,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(object_id, broker)
                )
            ''')

            # Galaxy information table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS galaxy_info (
                    id INTEGER PRIMARY KEY,
                    ra REAL NOT NULL,
                    dec REAL NOT NULL,
                    morphology TEXT,
                    catalog JSON,
                    redshift REAL,
                    magnitude_g REAL,
                    magnitude_r REAL,
                    magnitude_i REAL,
                    magnitude_z REAL,
                    extinction_json TEXT,
                    ned_name TEXT,
                    ned_separation_arcsec REAL,
                    queried_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ra, dec)
                )
            ''')

            # Add columns to existing galaxy_info tables that lack them
            for col_def in [
                ('extinction_json', 'TEXT'),
                ('ned_name', 'TEXT'),
                ('ned_separation_arcsec', 'REAL'),
            ]:
                try:
                    cursor.execute(
                        f'ALTER TABLE galaxy_info ADD COLUMN {col_def[0]} {col_def[1]}'
                    )
                except sqlite3.OperationalError:
                    pass  # column already exists

            # Merged alerts table (for deduplicated results)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS merged_alerts (
                    id INTEGER PRIMARY KEY,
                    unique_id TEXT UNIQUE,
                    ra REAL,
                    dec REAL,
                    discovery_date TEXT,
                    host_morphology TEXT,
                    brokers_detected TEXT,  -- comma-separated broker names
                    classification_antares JSON,
                    classification_alerce JSON,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Peak-fit targets for Magellan follow-up planning
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS peak_fit_targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id TEXT NOT NULL,
                    object_id_alerce_lsst TEXT,
                    object_id_antares TEXT,
                    rubin_dia_object_id TEXT,
                    ddf_field TEXT,
                    ra REAL NOT NULL,
                    dec REAL NOT NULL,
                    mean_ia_prob REAL,
                    brokers_detected TEXT,
                    peak_mjd REAL,
                    peak_mag REAL,
                    peak_mag_err REAL,
                    peak_band TEXT,
                    peak_fit_status TEXT,
                    mjd_now REAL,
                    delta_t REAL,
                    merit REAL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(object_id)
                )
            ''')

            conn.commit()
            logger.info(f"Initialized cache database at {self.db_path}")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def cache_alerts(self, broker: str, alerts_df: pd.DataFrame):
        """
        Cache alerts from a broker.

        Args:
            broker: Broker name
            alerts_df: DataFrame with alert data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for _, row in alerts_df.iterrows():
                # Separate numeric and non-numeric fields for JSON storage
                row_dict = row.to_dict()
                numeric_vals = {k: v for k, v in row_dict.items()
                                if isinstance(v, (int, float, np.integer, np.floating))
                                and pd.notna(v)}
                other_vals = {k: str(v) for k, v in row_dict.items()
                              if k not in numeric_vals and v is not None and pd.notna(v)}

                # Guard against NaN object_id — str(NaN) produces "nan"
                # which collides on UNIQUE constraint for multiple rows
                raw_oid = row.get('object_id', '')
                if pd.isna(raw_oid) or str(raw_oid).strip().lower() == 'nan':
                    logger.debug("Skipping alert with NaN object_id (ra=%.4f, dec=%.4f)",
                                 row.get('ra', 0), row.get('dec', 0))
                    continue

                cursor.execute('''
                    INSERT OR REPLACE INTO alerts
                    (object_id, broker, ra, dec, discovery_date, classification, metadata, cached_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    str(raw_oid),
                    broker,
                    float(row.get('ra', 0)),
                    float(row.get('dec', 0)),
                    str(row.get('discovery_date', '')),
                    json.dumps(numeric_vals, default=str),
                    json.dumps(other_vals, default=str)
                ))

            conn.commit()
            logger.info(f"Cached {len(alerts_df)} alerts from {broker}")

        except Exception as e:
            logger.error(f"Error caching alerts: {e}")

    def get_cached_alerts(self, broker: Optional[str] = None,
                         hours_old: int = 24) -> Optional[pd.DataFrame]:
        """
        Retrieve cached alerts.

        Args:
            broker: Specific broker name (None for all)
            hours_old: Only return data newer than this many hours

        Returns:
            DataFrame or None if no cached data
        """
        try:
            conn = sqlite3.connect(self.db_path)

            cutoff_time = datetime.now() - timedelta(hours=hours_old)

            if broker:
                query = f'''
                    SELECT * FROM alerts
                    WHERE broker = ? AND cached_at > '{cutoff_time.isoformat()}'
                '''
                df = pd.read_sql_query(query, conn, params=(broker,))
            else:
                query = f'''
                    SELECT * FROM alerts
                    WHERE cached_at > '{cutoff_time.isoformat()}'
                '''
                df = pd.read_sql_query(query, conn)

            if len(df) > 0:
                # Unpack classification and metadata JSON blobs back into columns
                df = self._unpack_json_columns(df)
                logger.info(f"Retrieved {len(df)} cached alerts")
                return df

            return None

        except Exception as e:
            logger.warning(f"Error retrieving cached alerts: {e}")
            return None

    def _unpack_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Unpack classification and metadata JSON blobs into top-level columns."""
        for json_col in ['classification', 'metadata']:
            if json_col not in df.columns:
                continue
            for idx, raw in df[json_col].items():
                if not raw or pd.isna(raw):
                    continue
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if k not in df.columns:
                                df[k] = np.nan
                            df.at[idx, k] = v
                except (json.JSONDecodeError, TypeError):
                    pass
        return df

    def cache_galaxy_info(self, ra: float, dec: float,
                         morphology: str, catalog_info: Dict[str, Any],
                         redshift: Optional[float] = None):
        """
        Cache galaxy information for coordinates.

        Args:
            ra: Right ascension
            dec: Declination
            morphology: Galaxy morphology classification
            catalog_info: Dictionary with catalog data
            redshift: Galaxy redshift
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            mags = catalog_info or {}
            cursor.execute('''
                INSERT OR REPLACE INTO galaxy_info
                (ra, dec, morphology, catalog, redshift, magnitude_g, magnitude_r, magnitude_i, magnitude_z, queried_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                ra, dec, morphology,
                json.dumps(catalog_info),
                redshift,
                mags.get('mag_g'),
                mags.get('mag_r'),
                mags.get('mag_i'),
                mags.get('mag_z')
            ))

            conn.commit()
            logger.info(f"Cached galaxy info for ({ra}, {dec})")

        except Exception as e:
            logger.warning(f"Error caching galaxy info: {e}")

    def get_cached_galaxy_info(self, ra: float, dec: float,
                              tolerance_arcmin: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached galaxy information.

        Args:
            ra: Right ascension
            dec: Declination
            tolerance_arcmin: Search tolerance in arcminutes

        Returns:
            Dictionary with galaxy info or None
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Simple tolerance in degrees
            tolerance_deg = tolerance_arcmin / 60.0

            query = f'''
                SELECT * FROM galaxy_info
                WHERE ABS(ra - {ra}) < {tolerance_deg}
                  AND ABS(dec - {dec}) < {tolerance_deg}
                LIMIT 1
            '''

            df = pd.read_sql_query(query, conn)

            if len(df) > 0:
                row = df.iloc[0]
                return {
                    'morphology': row.get('morphology'),
                    'catalog': json.loads(row.get('catalog', '{}')),
                    'redshift': row.get('redshift'),
                    'mag_g': row.get('magnitude_g'),
                    'mag_r': row.get('magnitude_r'),
                    'mag_i': row.get('magnitude_i'),
                    'mag_z': row.get('magnitude_z'),
                }

            return None

        except Exception as e:
            logger.warning(f"Error retrieving cached galaxy info: {e}")
            return None

    def cache_extinction(self, ra: float, dec: float, extinction_dict: Dict[str, float]):
        """Cache galactic extinction values for coordinates.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            extinction_dict: Mapping of band letter to A_SFD value
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            ext_json = json.dumps(extinction_dict)

            # Try to update existing row first
            cursor.execute('''
                UPDATE galaxy_info SET extinction_json = ?, queried_at = CURRENT_TIMESTAMP
                WHERE ABS(ra - ?) < 0.002 AND ABS(dec - ?) < 0.002
            ''', (ext_json, ra, dec))

            if cursor.rowcount == 0:
                # No existing row — insert new one
                cursor.execute('''
                    INSERT OR REPLACE INTO galaxy_info
                    (ra, dec, extinction_json, queried_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (ra, dec, ext_json))

            conn.commit()

        except Exception as e:
            logger.debug("Error caching extinction: %s", e)

    def get_cached_extinction(self, ra: float, dec: float,
                              tolerance_arcmin: float = 0.1) -> Optional[Dict[str, float]]:
        """Retrieve cached extinction values for coordinates.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            tolerance_arcmin: Search tolerance in arcminutes

        Returns:
            Dict of {band: A_SFD} or None if not cached
        """
        try:
            conn = sqlite3.connect(self.db_path)
            tolerance_deg = tolerance_arcmin / 60.0

            cursor = conn.cursor()
            cursor.execute('''
                SELECT extinction_json FROM galaxy_info
                WHERE ABS(ra - ?) < ? AND ABS(dec - ?) < ?
                  AND extinction_json IS NOT NULL
                LIMIT 1
            ''', (ra, tolerance_deg, dec, tolerance_deg))

            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return None

        except Exception as e:
            logger.debug("Error retrieving cached extinction: %s", e)
            return None

    def cache_ned_info(self, ra: float, dec: float, redshift: float,
                       ned_name: str = '', separation_arcsec: float = 0.0):
        """Cache NED redshift lookup results for coordinates.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            redshift: Spectroscopic redshift from NED
            ned_name: NED source name
            separation_arcsec: Angular separation to NED source
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Try to update existing row
            cursor.execute('''
                UPDATE galaxy_info
                SET redshift = ?, ned_name = ?, ned_separation_arcsec = ?,
                    queried_at = CURRENT_TIMESTAMP
                WHERE ABS(ra - ?) < 0.002 AND ABS(dec - ?) < 0.002
            ''', (redshift, ned_name, separation_arcsec, ra, dec))

            if cursor.rowcount == 0:
                cursor.execute('''
                    INSERT OR REPLACE INTO galaxy_info
                    (ra, dec, redshift, ned_name, ned_separation_arcsec, queried_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (ra, dec, redshift, ned_name, separation_arcsec))

            conn.commit()

        except Exception as e:
            logger.debug("Error caching NED info: %s", e)

    def get_cached_ned_info(self, ra: float, dec: float,
                            tolerance_arcmin: float = 0.1) -> Optional[Dict[str, Any]]:
        """Retrieve cached NED redshift info for coordinates.

        Returns:
            Dict with redshift, ned_name, ned_separation_arcsec, or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            tolerance_deg = tolerance_arcmin / 60.0

            cursor = conn.cursor()
            cursor.execute('''
                SELECT redshift, ned_name, ned_separation_arcsec FROM galaxy_info
                WHERE ABS(ra - ?) < ? AND ABS(dec - ?) < ?
                  AND redshift IS NOT NULL
                LIMIT 1
            ''', (ra, tolerance_deg, dec, tolerance_deg))

            row = cursor.fetchone()
            if row and row[0] is not None:
                return {
                    'redshift': row[0],
                    'ned_name': row[1] or '',
                    'ned_separation_arcsec': row[2] or 0.0,
                }
            return None

        except Exception as e:
            logger.debug("Error retrieving cached NED info: %s", e)
            return None

    def cache_merged_alerts(self, merged_df: pd.DataFrame):
        """
        Cache merged/deduplicated alerts.

        Args:
            merged_df: DataFrame with merged alert data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for _, row in merged_df.iterrows():
                raw_uid = row.get('unique_id', '')
                if pd.isna(raw_uid) or str(raw_uid).strip().lower() == 'nan':
                    logger.debug("Skipping merged alert with NaN unique_id (ra=%.4f, dec=%.4f)",
                                 row.get('ra', 0), row.get('dec', 0))
                    continue

                cursor.execute('''
                    INSERT OR REPLACE INTO merged_alerts
                    (unique_id, ra, dec, discovery_date, host_morphology, brokers_detected,
                     classification_antares, classification_alerce, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    str(raw_uid),
                    float(row.get('ra', 0)),
                    float(row.get('dec', 0)),
                    str(row.get('discovery_date', '')),
                    str(row.get('host_morphology', 'unknown')),
                    str(row.get('brokers_detected', '')),
                    json.dumps(row.get('classification_antares', {})),
                    json.dumps(row.get('classification_alerce', {}))
                ))

            conn.commit()
            logger.info(f"Cached {len(merged_df)} merged alerts")

        except Exception as e:
            logger.error(f"Error caching merged alerts: {e}")

    def get_cached_merged_alerts(self, hours_old: int = 24) -> Optional[pd.DataFrame]:
        """
        Retrieve cached merged alerts.

        Args:
            hours_old: Only return data newer than this many hours

        Returns:
            DataFrame or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_time = datetime.now() - timedelta(hours=hours_old)

            query = f'''
                SELECT * FROM merged_alerts
                WHERE last_updated > '{cutoff_time.isoformat()}'
            '''

            df = pd.read_sql_query(query, conn)

            if len(df) > 0:
                logger.info(f"Retrieved {len(df)} cached merged alerts")
                return df

            return None

        except Exception as e:
            logger.warning(f"Error retrieving cached merged alerts: {e}")
            return None

    def cache_peak_fit_targets(self, targets_df: pd.DataFrame):
        """Cache peak-fit targets with merit scores for Magellan planning.

        Replaces existing rows keyed on object_id.
        """
        cols = [
            'object_id', 'object_id_alerce_lsst', 'object_id_antares',
            'rubin_dia_object_id', 'ddf_field', 'ra', 'dec',
            'mean_ia_prob', 'brokers_detected',
            'peak_mjd', 'peak_mag', 'peak_mag_err', 'peak_band',
            'peak_fit_status', 'mjd_now', 'delta_t', 'merit',
        ]

        # Map DataFrame columns that may use different naming
        col_map = {
            'object_id_ALeRCE-LSST': 'object_id_alerce_lsst',
            'object_id_ANTARES': 'object_id_antares',
        }

        df = targets_df.copy()
        for src, dst in col_map.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

        # Resolve primary object_id if not present
        if 'object_id' not in df.columns:
            df['object_id'] = df.apply(
                lambda r: next(
                    (v for v in (r.get('object_id_alerce_lsst'),
                                 r.get('object_id_antares'),
                                 r.get('object_id_ALeRCE-LSST'),
                                 r.get('object_id_ANTARES'),
                                 r.get('object_id_ALeRCE'))
                     if pd.notna(v)),
                    f'idx_{r.name}'),
                axis=1)

        # Drop rows whose object_id is NaN or the literal string "nan"
        before = len(df)
        df = df[df['object_id'].apply(
            lambda v: pd.notna(v) and str(v).strip().lower() != 'nan'
        )].copy()
        if len(df) < before:
            logger.warning("Dropped %d rows with NaN object_id before caching",
                           before - len(df))

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Clear existing rows
            cursor.execute('DELETE FROM peak_fit_targets')

            for _, row in df.iterrows():
                vals = []
                for c in cols:
                    v = row.get(c)
                    # Catch float NaN, None, and the string "nan"
                    if isinstance(v, (float, np.floating)):
                        vals.append(None if pd.isna(v) else v)
                    elif v is None or (isinstance(v, str) and v.strip().lower() == 'nan'):
                        vals.append(None)
                    else:
                        vals.append(v)

                placeholders = ', '.join(['?'] * len(cols))
                cursor.execute(
                    f'INSERT OR REPLACE INTO peak_fit_targets '
                    f'({", ".join(cols)}) VALUES ({placeholders})',
                    vals)

            conn.commit()
            logger.info("Cached %d peak-fit targets for Magellan planning",
                        len(df))

        except Exception as e:
            logger.error("Error caching peak-fit targets: %s", e)

    def get_peak_fit_targets(self) -> Optional[pd.DataFrame]:
        """Retrieve cached peak-fit targets.

        Returns DataFrame or None if table is empty.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('SELECT * FROM peak_fit_targets', conn)
            if len(df) > 0:
                logger.info("Retrieved %d peak-fit targets", len(df))
                return df
            return None
        except Exception as e:
            logger.warning("Error retrieving peak-fit targets: %s", e)
            return None

    def clear_old_cache(self, days_old: int = 7):
        """
        Remove cache entries older than specified days.

        Args:
            days_old: Remove entries older than this many days
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_time = datetime.now() - timedelta(days=days_old)

            cursor.execute(f'''
                DELETE FROM alerts WHERE cached_at < '{cutoff_time.isoformat()}'
            ''')

            cursor.execute(f'''
                DELETE FROM galaxy_info WHERE queried_at < '{cutoff_time.isoformat()}'
            ''')

            conn.commit()
            logger.info(f"Cleared cache entries older than {days_old} days")

        except Exception as e:
            logger.warning(f"Error clearing old cache: {e}")
