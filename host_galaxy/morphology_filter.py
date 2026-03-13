"""Host galaxy morphology classification and filtering."""

import logging
from typing import Optional, Dict, Any
import pandas as pd

from utils.coordinates import CoordinateUtils
from utils.catalog_query import CatalogQuery
from cache.alert_cache import AlertCache

logger = logging.getLogger(__name__)


class MorphologyFilter:
    """Filter alerts based on host galaxy morphology."""

    def __init__(self, cache_dir: str = './cache/data'):
        """
        Initialize morphology filter.

        Args:
            cache_dir: Directory for caching galaxy data
        """
        self.cache = AlertCache(cache_dir)

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
            logger.info(f"Using cached galaxy info for ({ra:.3f}, {dec:.3f})")
            return cached_galaxy

        # Query catalogs
        morphology = 'unknown'
        galaxy_info = None

        # Try SDSS first (northern sky, dec > -10)
        if dec > -10:
            galaxy_info = CatalogQuery.query_sdss(ra, dec)
            if galaxy_info is None:
                # Try Pan-STARRS (dec > -30)
                galaxy_info = CatalogQuery.query_panstarrs(ra, dec)

        # Try SkyMapper for southern sky (dec < +10)
        if galaxy_info is None and dec < 10:
            galaxy_info = CatalogQuery.query_skymapper(ra, dec)

        # Log which catalog was used
        if galaxy_info:
            logger.debug("Host galaxy found in %s for (%.4f, %.4f)",
                        galaxy_info.get('catalog', '?'), ra, dec)
        else:
            logger.debug("No host galaxy found for (%.4f, %.4f) in any catalog", ra, dec)

        if galaxy_info:
            morphology = CatalogQuery.classify_morphology(galaxy_info)
            redshift = galaxy_info.get('redshift')

            # Cache the result
            self.cache.cache_galaxy_info(
                ra, dec, morphology,
                galaxy_info,
                redshift
            )

            logger.info(f"Classified host galaxy at ({ra:.3f}, {dec:.3f}) as {morphology}")

            return {
                'morphology': morphology,
                'catalog': galaxy_info,
                'redshift': redshift,
                'mag_g': galaxy_info.get('mag_g'),
                'mag_r': galaxy_info.get('mag_r'),
                'mag_i': galaxy_info.get('mag_i'),
                'mag_z': galaxy_info.get('mag_z'),
            }

        logger.warning(f"Could not classify host galaxy at ({ra:.3f}, {dec:.3f})")
        return {
            'morphology': 'unknown',
            'catalog': None,
            'redshift': None,
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
