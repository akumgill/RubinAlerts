"""NED (NASA/IPAC Extragalactic Database) redshift lookups.

Queries NED for spectroscopic redshifts of host galaxies near transient
candidates. Results are cached in the SQLite database.
"""

import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u
from astropy.cosmology import WMAP7

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_RADIUS_ARCSEC = 18.0  # 0.005 deg, matching ALeRCE notebook


def query_ned_redshift(ra: float, dec: float,
                       radius_arcsec: float = DEFAULT_SEARCH_RADIUS_ARCSEC
                       ) -> Optional[Dict[str, Any]]:
    """Query NED for the nearest object with a spectroscopic redshift.

    Parameters
    ----------
    ra, dec : float
        J2000 coordinates in decimal degrees.
    radius_arcsec : float
        Search radius in arcseconds.

    Returns
    -------
    Dict with keys: redshift, ned_name, separation_arcsec, distmod.
    Returns None if no object with redshift found or on failure.
    """
    try:
        from astroquery.ipac.ned import Ned
    except ImportError:
        logger.warning("astroquery.ipac.ned not available")
        return None

    try:
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='fk5')
        radius = radius_arcsec * u.arcsec

        result_table = Ned.query_region(coord, radius=radius, equinox='J2000.0')

        if result_table is None or len(result_table) == 0:
            return None

        df = result_table.to_pandas()

        # Filter to objects with valid redshifts
        has_z = df['Redshift'].notna()
        if not has_z.any():
            return None

        # Select closest object with valid redshift
        df_with_z = df[has_z]
        closest_idx = df_with_z['Separation'].idxmin()
        row = df_with_z.loc[closest_idx]

        redshift = float(row['Redshift'])
        ned_name = str(row.get('Object Name', ''))
        separation = float(row.get('Separation', 0.0))

        # Compute distance modulus
        distmod = np.nan
        if redshift > 0.001:  # avoid nonsensical values for very nearby objects
            try:
                distmod = float(Distance(z=redshift, cosmology=WMAP7).distmod / u.mag)
            except Exception:
                pass

        return {
            'redshift': redshift,
            'ned_name': ned_name,
            'separation_arcsec': separation,
            'distmod': distmod,
        }

    except Exception as e:
        logger.debug("NED query failed for (%.4f, %.4f): %s", ra, dec, e)
        return None


def query_ned_batch(coords_df: pd.DataFrame,
                    cache=None,
                    radius_arcsec: float = DEFAULT_SEARCH_RADIUS_ARCSEC
                    ) -> pd.DataFrame:
    """Batch NED redshift lookups for a DataFrame of coordinates.

    Parameters
    ----------
    coords_df : pd.DataFrame
        Must contain 'ra' and 'dec' columns.
    cache : AlertCache, optional
        If provided, checks/stores results in SQLite.
    radius_arcsec : float
        Search radius for NED queries.

    Returns
    -------
    Copy of coords_df with added columns: ned_redshift, ned_distmod,
        ned_name, ned_sep_arcsec.
    """
    df = coords_df.copy()
    df['ned_redshift'] = np.nan
    df['ned_distmod'] = np.nan
    df['ned_name'] = ''
    df['ned_sep_arcsec'] = np.nan

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
            cached = cache.get_cached_ned_info(ra, dec)
            if cached is not None:
                df.at[idx, 'ned_redshift'] = cached['redshift']
                df.at[idx, 'ned_name'] = cached.get('ned_name', '')
                df.at[idx, 'ned_sep_arcsec'] = cached.get('ned_separation_arcsec', np.nan)
                # Recompute distmod from cached redshift
                z = cached['redshift']
                if z > 0.001:
                    try:
                        df.at[idx, 'ned_distmod'] = float(
                            Distance(z=z, cosmology=WMAP7).distmod / u.mag
                        )
                    except Exception:
                        pass
                n_cached += 1
                continue

        # Query NED
        result = query_ned_redshift(ra, dec, radius_arcsec=radius_arcsec)

        if result is not None:
            df.at[idx, 'ned_redshift'] = result['redshift']
            df.at[idx, 'ned_distmod'] = result['distmod']
            df.at[idx, 'ned_name'] = result['ned_name']
            df.at[idx, 'ned_sep_arcsec'] = result['separation_arcsec']
            n_queried += 1

            # Store in cache
            if cache is not None:
                try:
                    cache.cache_ned_info(
                        ra, dec,
                        redshift=result['redshift'],
                        ned_name=result['ned_name'],
                        separation_arcsec=result['separation_arcsec'],
                    )
                except Exception as e:
                    logger.debug("Failed to cache NED info: %s", e)
        else:
            n_failed += 1

        # Progress logging every 25 objects (NED is slower than IRSA)
        done = n_cached + n_queried + n_failed
        if done % 25 == 0 and done > 0:
            logger.info("NED lookup: %d/%d done (%d cached, %d queried, %d no result)",
                        done, total, n_cached, n_queried, n_failed)

    logger.info("NED lookup complete: %d total (%d cached, %d queried, %d no result)",
                total, n_cached, n_queried, n_failed)
    return df
