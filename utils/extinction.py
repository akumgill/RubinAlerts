"""Galactic dust extinction corrections using IRSA Dust maps.

Queries the SFD (Schlegel, Finkbeiner & Davis 1998) dust maps via
astroquery to obtain galactic extinction A_lambda for each photometric band.
Results are cached in the SQLite database to avoid repeated IRSA queries.
"""

import json
import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

logger = logging.getLogger(__name__)

# IRSA dust map filter names → our short band keys
IRSA_FILTER_MAP = {
    'SDSS u': 'u',
    'SDSS g': 'g',
    'SDSS r': 'r',
    'SDSS i': 'i',
    'SDSS z': 'z',
}


def get_extinction(ra: float, dec: float) -> Dict[str, float]:
    """Query IRSA for SFD galactic extinction at a sky position.

    Parameters
    ----------
    ra, dec : float
        J2000 coordinates in decimal degrees.

    Returns
    -------
    dict mapping band letter ('u','g','r','i','z') to A_SFD in magnitudes.
    Returns empty dict on failure.
    """
    try:
        from astroquery.ipac.irsa.irsa_dust import IrsaDust
    except ImportError:
        logger.warning("astroquery.ipac.irsa.irsa_dust not available")
        return {}

    try:
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='fk5')
        table = IrsaDust.get_extinction_table(coord)

        extinction = {}
        for irsa_name, band_key in IRSA_FILTER_MAP.items():
            mask = table['Filter_name'] == irsa_name
            if mask.any():
                extinction[band_key] = float(table[mask]['A_SFD'][0])

        return extinction

    except Exception as e:
        logger.warning("IRSA dust query failed for (%.4f, %.4f): %s", ra, dec, e)
        return {}


def get_extinction_batch(coords_df: pd.DataFrame,
                         cache=None) -> pd.DataFrame:
    """Fetch galactic extinction for a DataFrame of coordinates.

    Parameters
    ----------
    coords_df : pd.DataFrame
        Must contain 'ra' and 'dec' columns (decimal degrees).
    cache : AlertCache, optional
        If provided, checks/stores extinction values in SQLite.

    Returns
    -------
    Copy of coords_df with added columns: A_u, A_g, A_r, A_i, A_z.
    """
    df = coords_df.copy()
    for band in ['u', 'g', 'r', 'i', 'z']:
        df[f'A_{band}'] = np.nan

    total = len(df)
    if total == 0:
        return df

    n_cached = 0
    n_queried = 0
    n_failed = 0

    for idx, row in df.iterrows():
        ra, dec = row['ra'], row['dec']
        if pd.isna(ra) or pd.isna(dec):
            continue

        # Try cache first
        if cache is not None:
            cached = cache.get_cached_extinction(ra, dec)
            if cached is not None:
                for band, a_val in cached.items():
                    if f'A_{band}' in df.columns:
                        df.at[idx, f'A_{band}'] = a_val
                n_cached += 1
                continue

        # Query IRSA
        extinction = get_extinction(ra, dec)

        if extinction:
            for band, a_val in extinction.items():
                if f'A_{band}' in df.columns:
                    df.at[idx, f'A_{band}'] = a_val
            n_queried += 1

            # Store in cache
            if cache is not None:
                try:
                    cache.cache_extinction(ra, dec, extinction)
                except Exception as e:
                    logger.debug("Failed to cache extinction: %s", e)
        else:
            n_failed += 1

        # Progress logging every 50 objects
        done = n_cached + n_queried + n_failed
        if done % 50 == 0 and done > 0:
            logger.info("Extinction lookup: %d/%d done (%d cached, %d queried, %d failed)",
                        done, total, n_cached, n_queried, n_failed)

    logger.info("Extinction lookup complete: %d total (%d cached, %d queried, %d failed)",
                total, n_cached, n_queried, n_failed)
    return df


def correct_magnitude(mag: float, A_band: float) -> float:
    """Apply galactic extinction correction to a magnitude.

    Parameters
    ----------
    mag : float
        Observed (uncorrected) magnitude.
    A_band : float
        Galactic extinction in this band (magnitudes).

    Returns
    -------
    Extinction-corrected magnitude (mag - A_band), or NaN if inputs invalid.
    """
    if pd.isna(mag) or pd.isna(A_band):
        return np.nan
    return mag - A_band
