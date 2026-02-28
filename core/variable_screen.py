"""Screen candidates against known variable star catalogs in Rubin DDFs."""

import os
import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Default catalog directory (relative to project root)
DEFAULT_CATALOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'variable_catalogs'
)

CATALOG_FILES = [
    'COSMOS_variables.csv',
    'XMM-LSS_variables.csv',
    'ECDFS_variables.csv',
    'ELAIS-S1_variables.csv',
    'EDFS_a_variables.csv',
    'EDFS_b_variables.csv',
]

MATCH_RADIUS_ARCSEC = 2.0


class VariableScreener:
    """Screen transient candidates against known variable source catalogs.

    Uses the compiled catalogs from astrostubbs/Rubin-Deep-Drilling-Variable-Catalogs
    which contain ~13,749 known variables across the 6 Rubin DDFs from 11 databases.
    """

    def __init__(self, catalog_dir: Optional[str] = None):
        self.catalog_dir = catalog_dir or DEFAULT_CATALOG_DIR
        self.variables_df = None
        self._load_catalogs()

    def _load_catalogs(self):
        """Load all DDF variable catalogs into a single DataFrame."""
        frames = []
        for fname in CATALOG_FILES:
            path = os.path.join(self.catalog_dir, fname)
            if not os.path.exists(path):
                logger.warning(f"Variable catalog not found: {path}")
                continue
            try:
                df = pd.read_csv(path)
                frames.append(df)
                logger.info(f"Loaded {len(df)} variables from {fname}")
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")

        if frames:
            self.variables_df = pd.concat(frames, ignore_index=True)
            # Ensure RA/Dec columns exist
            if 'RA' not in self.variables_df.columns or 'Dec' not in self.variables_df.columns:
                logger.error("Variable catalogs missing RA/Dec columns")
                self.variables_df = None
                return
            self.variables_df = self.variables_df.dropna(subset=['RA', 'Dec'])
            logger.info(f"Loaded {len(self.variables_df)} known variables total")
        else:
            logger.warning("No variable catalogs loaded. "
                          "Download from github.com/astrostubbs/Rubin-Deep-Drilling-Variable-Catalogs")
            self.variables_df = None

    def screen_candidates(self, candidates_df: pd.DataFrame,
                          match_radius_arcsec: float = MATCH_RADIUS_ARCSEC) -> pd.DataFrame:
        """Cross-match candidates against known variables.

        Adds columns:
        - known_variable: bool
        - variable_type: str (Otype from catalog)
        - variable_id: str (Identifier from catalog)

        Args:
            candidates_df: DataFrame with 'ra' and 'dec' columns
            match_radius_arcsec: Cross-match radius in arcseconds

        Returns:
            Annotated DataFrame with variable screening columns
        """
        df = candidates_df.copy()
        df['known_variable'] = False
        df['variable_type'] = ''
        df['variable_id'] = ''

        if self.variables_df is None or len(self.variables_df) == 0:
            logger.warning("No variable catalogs available; skipping screening")
            return df

        if 'ra' not in df.columns or 'dec' not in df.columns:
            logger.warning("Candidates missing ra/dec; skipping screening")
            return df

        tol_deg = match_radius_arcsec / 3600.0
        var_ra = self.variables_df['RA'].values
        var_dec = self.variables_df['Dec'].values

        n_matched = 0
        for idx, row in df.iterrows():
            ra = row['ra']
            dec = row['dec']

            # Simple rectangular match (fast approximation; good enough at < 2")
            cos_dec = np.cos(np.radians(dec))
            ra_diff = np.abs(var_ra - ra) * cos_dec
            dec_diff = np.abs(var_dec - dec)
            matches = (ra_diff < tol_deg) & (dec_diff < tol_deg)

            if matches.any():
                match_idx = np.where(matches)[0][0]
                match_row = self.variables_df.iloc[match_idx]
                df.at[idx, 'known_variable'] = True
                df.at[idx, 'variable_type'] = str(match_row.get('Otype', ''))
                df.at[idx, 'variable_id'] = str(match_row.get('Identifier', ''))
                n_matched += 1

        logger.info(f"Variable screening: {n_matched}/{len(df)} candidates "
                   f"match known variables (within {match_radius_arcsec}\")")
        return df
