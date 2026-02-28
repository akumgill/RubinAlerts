"""Galaxy catalog queries and cross-matching utilities."""

import logging
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class CatalogQuery:
    """Query astronomical catalogs for galaxy information."""

    @staticmethod
    def query_sdss(ra: float, dec: float, radius_arcmin: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Query SDSS for galaxy at given coordinates.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius_arcmin: Search radius in arcminutes

        Returns:
            Dictionary with galaxy properties or None
        """
        try:
            from astroquery.sdss import SDSS
            from astropy.coordinates import SkyCoord
            from astropy import units as u

            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

            # Query SDSS for photometric objects
            result = SDSS.query_region(
                coord,
                radius=radius_arcmin*u.arcmin,
                photoobj_fields=['objid', 'ra', 'dec', 'modelMag_g', 'modelMag_r',
                                'modelMag_i', 'modelMag_z', 'extinction_r',
                                'specClass', 'z', 'zErr', 'type']
            )

            if result is not None and len(result) > 0:
                # Sort by closest to search coordinates
                result['separation'] = [
                    SkyCoord(ra=r['ra']*u.deg, dec=r['dec']*u.deg).separation(coord).arcmin
                    for r in result
                ]
                closest = result[result['separation'] == result['separation'].min()][0]

                return {
                    'catalog': 'SDSS',
                    'objid': closest['objid'],
                    'ra': closest['ra'],
                    'dec': closest['dec'],
                    'mag_g': closest['modelMag_g'],
                    'mag_r': closest['modelMag_r'],
                    'mag_i': closest['modelMag_i'],
                    'mag_z': closest['modelMag_z'],
                    'extinction_r': closest['extinction_r'],
                    'redshift': closest['z'],
                    'z_err': closest['zErr'],
                    'spec_class': closest['specClass'],
                    'type': closest['type'],
                    'separation_arcmin': closest['separation']
                }
            return None

        except Exception as e:
            logger.warning(f"Failed to query SDSS: {e}")
            return None

    @staticmethod
    def query_panstarrs(ra: float, dec: float, radius_arcmin: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Query Pan-STARRS for galaxy at given coordinates.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius_arcmin: Search radius in arcminutes

        Returns:
            Dictionary with galaxy properties or None
        """
        try:
            from astroquery.mast import Catalogs
            from astropy.coordinates import SkyCoord
            from astropy import units as u

            # Pan-STARRS catalog query
            result = Catalogs.query_region(
                coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg),
                catalog='Panstarrs',
                data_release='dr2',
                radius=radius_arcmin*u.arcmin
            )

            if result is not None and len(result) > 0:
                coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                result['separation'] = [
                    SkyCoord(ra=r['ra']*u.deg, dec=r['dec']*u.deg).separation(coord).arcmin
                    for r in result
                ]
                closest = result[result['separation'] == result['separation'].min()][0]

                return {
                    'catalog': 'Pan-STARRS',
                    'objid': closest['objID'],
                    'ra': closest['ra'],
                    'dec': closest['dec'],
                    'mag_g': closest['gMeanPSFMag'],
                    'mag_r': closest['rMeanPSFMag'],
                    'mag_i': closest['iMeanPSFMag'],
                    'mag_z': closest['zMeanPSFMag'],
                    'mag_w1': closest['w1MeanPSFMag'] if 'w1MeanPSFMag' in closest.colnames else None,
                    'separation_arcmin': closest['separation']
                }
            return None

        except Exception as e:
            logger.warning(f"Failed to query Pan-STARRS: {e}")
            return None

    @staticmethod
    def classify_morphology(galaxy_info: Dict[str, Any]) -> str:
        """
        Classify galaxy morphology based on color and magnitude properties.

        Args:
            galaxy_info: Dictionary with galaxy properties (magnitudes, etc.)

        Returns:
            Morphology classification: 'elliptical', 'spiral', 'uncertain', 'unknown'
        """
        try:
            # Simple morphology classification based on color and concentration
            # Early-type (elliptical) galaxies are typically:
            # - Red (g-r > 0.6, r-i > 0.2)
            # - High surface brightness concentration

            if not galaxy_info:
                return 'unknown'

            mag_g = galaxy_info.get('mag_g')
            mag_r = galaxy_info.get('mag_r')
            mag_i = galaxy_info.get('mag_i')

            if mag_g is None or mag_r is None or mag_i is None:
                return 'uncertain'

            # Calculate colors
            g_r = mag_g - mag_r
            r_i = mag_r - mag_i

            # Red sequence (early-type) galaxies
            if g_r > 0.55 and r_i > 0.15:
                return 'elliptical'

            # Blue cloud (late-type) galaxies
            elif g_r < 0.45 and r_i < 0.15:
                return 'spiral'

            else:
                return 'uncertain'

        except Exception as e:
            logger.warning(f"Error classifying morphology: {e}")
            return 'unknown'
