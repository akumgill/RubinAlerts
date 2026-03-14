"""Host galaxy morphology classification and filtering."""

import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from utils.coordinates import CoordinateUtils
from utils.catalog_query import CatalogQuery
from cache.alert_cache import AlertCache

logger = logging.getLogger(__name__)

# Nuclear offset thresholds (arcseconds)
# Objects within this radius of host center are flagged as potential nuclear sources
NUCLEAR_OFFSET_THRESHOLD_ARCSEC = 1.0  # AGN/TDE typically < 0.5-1"
# SNe are typically offset by several arcsec, but still within host
MAX_SN_OFFSET_ARCSEC = 30.0  # Beyond this, likely not associated


class MorphologyFilter:
    """Filter alerts based on host galaxy morphology."""

    def __init__(self, cache_dir: str = './cache/data'):
        """
        Initialize morphology filter.

        Args:
            cache_dir: Directory for caching galaxy data
        """
        self.cache = AlertCache(cache_dir)

    @staticmethod
    def compute_angular_separation(ra1: float, dec1: float,
                                   ra2: float, dec2: float) -> float:
        """Compute angular separation between two sky positions.

        Parameters
        ----------
        ra1, dec1 : float
            First position (degrees)
        ra2, dec2 : float
            Second position (degrees)

        Returns
        -------
        Separation in arcseconds
        """
        # Convert to radians
        ra1_rad = np.radians(ra1)
        dec1_rad = np.radians(dec1)
        ra2_rad = np.radians(ra2)
        dec2_rad = np.radians(dec2)

        # Haversine formula for small angles
        cos_sep = (np.sin(dec1_rad) * np.sin(dec2_rad) +
                   np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(ra1_rad - ra2_rad))
        cos_sep = np.clip(cos_sep, -1, 1)  # Handle numerical precision
        sep_rad = np.arccos(cos_sep)

        # Convert to arcseconds
        return np.degrees(sep_rad) * 3600.0

    def compute_nuclear_offset(self, transient_ra: float, transient_dec: float,
                               host_info: Dict[str, Any]) -> Tuple[float, str]:
        """Compute offset between transient and host galaxy nucleus.

        Parameters
        ----------
        transient_ra, transient_dec : float
            Transient position (degrees)
        host_info : dict
            Host galaxy info from classify_host_galaxy (must have 'catalog' with 'ra', 'dec')

        Returns
        -------
        (offset_arcsec, classification) tuple where classification is:
        - 'nuclear': offset < 1" (likely AGN/TDE)
        - 'offset': 1" < offset < 30" (consistent with SN)
        - 'distant': offset > 30" (may not be associated)
        - 'unknown': no host position available
        """
        catalog = host_info.get('catalog')
        if not catalog:
            return np.nan, 'unknown'

        host_ra = catalog.get('ra')
        host_dec = catalog.get('dec')

        if host_ra is None or host_dec is None:
            return np.nan, 'unknown'

        offset_arcsec = self.compute_angular_separation(
            transient_ra, transient_dec, host_ra, host_dec
        )

        if offset_arcsec < NUCLEAR_OFFSET_THRESHOLD_ARCSEC:
            return offset_arcsec, 'nuclear'
        elif offset_arcsec < MAX_SN_OFFSET_ARCSEC:
            return offset_arcsec, 'offset'
        else:
            return offset_arcsec, 'distant'

    def classify_host_galaxy(self, ra: float, dec: float) -> Dict[str, Any]:
        """
        Classify host galaxy morphology for given coordinates.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)

        Returns:
            Dictionary with morphology classification and galaxy properties
        """
        # Check cache first
        cached_galaxy = self.cache.get_cached_galaxy_info(ra, dec)
        if cached_galaxy:
            logger.debug(f"Using cached galaxy info for ({ra:.3f}, {dec:.3f})")
            # Compute nuclear offset if not already present (for backwards compat with old cache)
            if 'nuclear_offset_arcsec' not in cached_galaxy or cached_galaxy.get('nuclear_offset_arcsec') is None:
                offset_arcsec, offset_class = self.compute_nuclear_offset(ra, dec, cached_galaxy)
                cached_galaxy['nuclear_offset_arcsec'] = offset_arcsec
                cached_galaxy['offset_class'] = offset_class
                # Also extract host coords from catalog if available
                cat = cached_galaxy.get('catalog', {})
                if isinstance(cat, dict):
                    cached_galaxy['host_ra'] = cat.get('ra')
                    cached_galaxy['host_dec'] = cat.get('dec')
            return cached_galaxy

        # Query catalogs
        morphology = 'unknown'
        galaxy_info = None

        # Search radius in arcmin - larger for host galaxy searches
        # since SNe can be offset from galaxy centers
        search_radius = 2.0  # arcmin

        # Try SDSS first (northern sky, dec > -10)
        if dec > -10:
            galaxy_info = CatalogQuery.query_sdss(ra, dec, radius_arcmin=search_radius)

        # Always try Pan-STARRS for dec > -30 (deeper than SDSS, good coverage)
        if galaxy_info is None and dec > -30:
            galaxy_info = CatalogQuery.query_panstarrs(ra, dec, radius_arcmin=search_radius)

        # Try SkyMapper for southern sky (dec < +10)
        if galaxy_info is None and dec < 10:
            galaxy_info = CatalogQuery.query_skymapper(ra, dec, radius_arcmin=search_radius)

        # Final fallback: GLADE+ galaxy catalog (all-sky, optimized for transients)
        if galaxy_info is None:
            galaxy_info = CatalogQuery.query_glade(ra, dec, radius_arcmin=search_radius)

        # Log which catalog was used
        if galaxy_info:
            logger.debug("Host galaxy found in %s for (%.4f, %.4f)",
                        galaxy_info.get('catalog', '?'), ra, dec)
        else:
            logger.debug("No host galaxy found for (%.4f, %.4f) in any catalog", ra, dec)

        if galaxy_info:
            morphology = CatalogQuery.classify_morphology(galaxy_info)
            redshift = galaxy_info.get('redshift')

            # Compute nuclear offset
            offset_arcsec, offset_class = self.compute_nuclear_offset(
                ra, dec, {'catalog': galaxy_info}
            )

            # Cache the result
            self.cache.cache_galaxy_info(
                ra, dec, morphology,
                galaxy_info,
                redshift
            )

            logger.info(f"Classified host galaxy at ({ra:.3f}, {dec:.3f}) as {morphology} "
                       f"(offset={offset_arcsec:.2f}\", class={offset_class})")

            return {
                'morphology': morphology,
                'catalog': galaxy_info,
                'redshift': redshift,
                'mag_g': galaxy_info.get('mag_g'),
                'mag_r': galaxy_info.get('mag_r'),
                'mag_i': galaxy_info.get('mag_i'),
                'mag_z': galaxy_info.get('mag_z'),
                'nuclear_offset_arcsec': offset_arcsec,
                'offset_class': offset_class,
                'host_ra': galaxy_info.get('ra'),
                'host_dec': galaxy_info.get('dec'),
            }

        logger.warning(f"Could not classify host galaxy at ({ra:.3f}, {dec:.3f})")
        return {
            'morphology': 'unknown',
            'catalog': None,
            'redshift': None,
            'nuclear_offset_arcsec': np.nan,
            'offset_class': 'unknown',
            'host_ra': None,
            'host_dec': None,
        }

    def filter_elliptical(self, alerts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter alerts for Type Ia supernovae in elliptical galaxies.

        Args:
            alerts_df: DataFrame with alert data (must include 'ra', 'dec')

        Returns:
            Filtered DataFrame with only elliptical galaxy hosts
        """
        if 'ra' not in alerts_df.columns or 'dec' not in alerts_df.columns:
            logger.error("Alert DataFrame must include 'ra' and 'dec' columns")
            return pd.DataFrame()

        results = []

        for _, alert in alerts_df.iterrows():
            try:
                galaxy_info = self.classify_host_galaxy(
                    float(alert['ra']),
                    float(alert['dec'])
                )

                if galaxy_info['morphology'] == 'elliptical':
                    # Add galaxy classification to alert
                    alert_copy = alert.copy()
                    alert_copy['host_morphology'] = galaxy_info['morphology']
                    alert_copy['host_redshift'] = galaxy_info.get('redshift')
                    alert_copy['host_mag_g'] = galaxy_info.get('mag_g')
                    alert_copy['host_mag_r'] = galaxy_info.get('mag_r')
                    alert_copy['host_mag_i'] = galaxy_info.get('mag_i')
                    alert_copy['host_mag_z'] = galaxy_info.get('mag_z')
                    results.append(alert_copy)

            except Exception as e:
                logger.warning(f"Error processing alert at ({alert['ra']}, {alert['dec']}): {e}")
                continue

        if results:
            filtered_df = pd.DataFrame(results)
            logger.info(f"Filtered to {len(filtered_df)} alerts in elliptical galaxies")
            return filtered_df
        else:
            logger.info("No alerts found in elliptical galaxies")
            return pd.DataFrame()
