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
            import numpy as np

            # Use SQL query (more reliable than query_region)
            # type=3 is galaxy in SDSS
            radius_deg = radius_arcmin / 60.0
            sql = f"""
            SELECT TOP 10
                objid, ra, dec,
                modelMag_g, modelMag_r, modelMag_i, modelMag_z,
                extinction_r, type
            FROM PhotoObj
            WHERE ra BETWEEN {ra - radius_deg} AND {ra + radius_deg}
              AND dec BETWEEN {dec - radius_deg} AND {dec + radius_deg}
              AND type = 3
            ORDER BY
                SQRT(POWER(ra - {ra}, 2) + POWER(dec - {dec}, 2))
            """

            result = SDSS.query_sql(sql)

            if result is not None and len(result) > 0:
                closest = result[0]
                coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                obj_coord = SkyCoord(ra=float(closest['ra'])*u.deg,
                                     dec=float(closest['dec'])*u.deg)
                separation = coord.separation(obj_coord).arcmin

                logger.debug("SDSS: found galaxy at %.4f, %.4f (sep=%.2f')",
                             float(closest['ra']), float(closest['dec']), separation)
                return {
                    'catalog': 'SDSS',
                    'objid': closest['objid'],
                    'ra': float(closest['ra']),
                    'dec': float(closest['dec']),
                    'mag_g': float(closest['modelMag_g']),
                    'mag_r': float(closest['modelMag_r']),
                    'mag_i': float(closest['modelMag_i']),
                    'mag_z': float(closest['modelMag_z']),
                    'extinction_r': float(closest['extinction_r']),
                    'redshift': None,  # Not in PhotoObj, would need SpecObj join
                    'z_err': None,
                    'spec_class': None,
                    'type': int(closest['type']),
                    'separation_arcmin': separation
                }
            logger.debug("SDSS: no galaxy within %.1f' of (%.4f, %.4f)",
                        radius_arcmin, ra, dec)
            return None

        except Exception as e:
            logger.debug("SDSS query failed for (%.4f, %.4f): %s", ra, dec, e)
            return None

    @staticmethod
    def query_panstarrs(ra: float, dec: float, radius_arcmin: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Query Pan-STARRS DR2 via VizieR for galaxy at given coordinates.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius_arcmin: Search radius in arcminutes

        Returns:
            Dictionary with galaxy properties or None
        """
        try:
            from astroquery.vizier import Vizier
            from astropy.coordinates import SkyCoord
            from astropy import units as u
            import numpy as np

            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

            # Query Pan-STARRS DR1 via VizieR (II/349/ps1)
            v = Vizier(columns=['objID', 'RAJ2000', 'DEJ2000',
                               'gmag', 'rmag', 'imag', 'zmag'],
                      row_limit=10)
            result = v.query_region(coord, radius=radius_arcmin*u.arcmin,
                                    catalog='II/349/ps1')

            if not result or len(result) == 0 or len(result[0]) == 0:
                return None

            table = result[0]

            # Find closest object
            best_sep = float('inf')
            best_row = None

            for row in table:
                try:
                    row_ra = float(row['RAJ2000'])
                    row_dec = float(row['DEJ2000'])
                    if not np.isfinite(row_ra) or not np.isfinite(row_dec):
                        continue
                    obj_coord = SkyCoord(ra=row_ra*u.deg, dec=row_dec*u.deg)
                    sep = coord.separation(obj_coord).arcmin
                    if sep < best_sep:
                        best_sep = sep
                        best_row = row
                except (ValueError, TypeError):
                    continue

            if best_row is None:
                return None

            def safe_float(val):
                try:
                    f = float(val)
                    return f if np.isfinite(f) else None
                except (ValueError, TypeError):
                    return None

            return {
                'catalog': 'Pan-STARRS',
                'objid': best_row['objID'],
                'ra': safe_float(best_row['RAJ2000']),
                'dec': safe_float(best_row['DEJ2000']),
                'mag_g': safe_float(best_row['gmag']),
                'mag_r': safe_float(best_row['rmag']),
                'mag_i': safe_float(best_row['imag']),
                'mag_z': safe_float(best_row['zmag']),
                'separation_arcmin': best_sep
            }

        except Exception as e:
            logger.warning(f"Failed to query Pan-STARRS: {e}")
            return None

    @staticmethod
    def query_skymapper(ra: float, dec: float, radius_arcmin: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Query SkyMapper DR4 via VizieR for southern sky galaxies.

        SkyMapper covers dec < +10, optimal for southern DDFs like ELAIS-S1, EDFS.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius_arcmin: Search radius in arcminutes

        Returns:
            Dictionary with galaxy properties or None
        """
        try:
            from astroquery.vizier import Vizier
            from astropy.coordinates import SkyCoord
            from astropy import units as u
            import numpy as np

            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

            # SkyMapper DR4 (II/379/smssdr4)
            v = Vizier(columns=['ObjectId', 'RAICRS', 'DEICRS',
                               'gPSF', 'rPSF', 'iPSF', 'zPSF', 'ClassStar'],
                      row_limit=10)
            result = v.query_region(coord, radius=radius_arcmin*u.arcmin,
                                    catalog='II/379/smssdr4')

            if not result or len(result) == 0 or len(result[0]) == 0:
                return None

            table = result[0]

            # Find closest galaxy (ClassStar < 0.5 = likely extended)
            best_sep = float('inf')
            best_row = None

            for row in table:
                try:
                    row_ra = float(row['RAICRS'])
                    row_dec = float(row['DEICRS'])
                    class_star = float(row['ClassStar']) if 'ClassStar' in row.colnames else 0
                    if not np.isfinite(row_ra) or not np.isfinite(row_dec):
                        continue
                    # Prefer extended objects (galaxies)
                    if class_star > 0.8:
                        continue
                    obj_coord = SkyCoord(ra=row_ra*u.deg, dec=row_dec*u.deg)
                    sep = coord.separation(obj_coord).arcmin
                    if sep < best_sep:
                        best_sep = sep
                        best_row = row
                except (ValueError, TypeError):
                    continue

            if best_row is None:
                return None

            def safe_float(val):
                try:
                    f = float(val)
                    return f if np.isfinite(f) else None
                except (ValueError, TypeError):
                    return None

            return {
                'catalog': 'SkyMapper',
                'objid': best_row['ObjectId'],
                'ra': safe_float(best_row['RAICRS']),
                'dec': safe_float(best_row['DEICRS']),
                'mag_g': safe_float(best_row['gPSF']),
                'mag_r': safe_float(best_row['rPSF']),
                'mag_i': safe_float(best_row['iPSF']),
                'mag_z': safe_float(best_row['zPSF']),
                'separation_arcmin': best_sep
            }

        except Exception as e:
            logger.warning(f"Failed to query SkyMapper: {e}")
            return None

    @staticmethod
    def query_glade(ra: float, dec: float, radius_arcmin: float = 2.0) -> Optional[Dict[str, Any]]:
        """
        Query GLADE+ galaxy catalog via VizieR for nearby galaxies.

        GLADE+ is optimized for gravitational wave follow-up and contains
        ~22 million galaxies with good completeness to ~300 Mpc.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            radius_arcmin: Search radius in arcminutes

        Returns:
            Dictionary with galaxy properties or None
        """
        try:
            from astroquery.vizier import Vizier
            from astropy.coordinates import SkyCoord
            from astropy import units as u
            import numpy as np

            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

            # Query GLADE+ catalog (VII/291)
            v = Vizier(columns=['GLADE+', 'RAJ2000', 'DEJ2000', 'zhelio', 'Bmag', 'Jmag', 'W1mag', 'W2mag'],
                      row_limit=10)
            result = v.query_region(coord, radius=radius_arcmin*u.arcmin,
                                    catalog='VII/291')

            if not result or len(result) == 0 or len(result[0]) == 0:
                return None

            table = result[0]

            # Find closest galaxy
            best_sep = float('inf')
            best_row = None

            for row in table:
                try:
                    row_ra = float(row['RAJ2000'])
                    row_dec = float(row['DEJ2000'])
                    if not np.isfinite(row_ra) or not np.isfinite(row_dec):
                        continue
                    obj_coord = SkyCoord(ra=row_ra*u.deg, dec=row_dec*u.deg)
                    sep = coord.separation(obj_coord).arcmin
                    if sep < best_sep:
                        best_sep = sep
                        best_row = row
                except (ValueError, TypeError):
                    continue

            if best_row is None:
                return None

            def safe_float(val):
                try:
                    f = float(val)
                    return f if np.isfinite(f) else None
                except (ValueError, TypeError):
                    return None

            # GLADE+ has B mag (some), J mag (some), and WISE W1/W2 (most)
            b_mag = safe_float(best_row['Bmag']) if 'Bmag' in best_row.colnames else None
            j_mag = safe_float(best_row['Jmag']) if 'Jmag' in best_row.colnames else None
            w1_mag = safe_float(best_row['W1mag']) if 'W1mag' in best_row.colnames else None
            w2_mag = safe_float(best_row['W2mag']) if 'W2mag' in best_row.colnames else None

            # Estimate optical mags from available photometry
            # If B available: g ≈ B - 0.3, r ≈ B - 0.5 for typical galaxies
            # If only WISE: rough estimate from W1 (not accurate for morphology)
            if b_mag is not None:
                mag_g = b_mag - 0.3
                mag_r = b_mag - 0.5
                mag_i = b_mag - 0.7
            elif w1_mag is not None and w2_mag is not None:
                # Very rough estimate - W1-W2 color can indicate galaxy type
                # Red galaxies: W1-W2 ~ 0, Blue galaxies: W1-W2 < 0
                w1w2 = w1_mag - w2_mag
                # Estimate g-r from W1-W2 (ellipticals have g-r > 0.6, spirals < 0.5)
                if w1w2 > -0.1:  # Likely red/elliptical
                    mag_g = None  # Can't reliably estimate
                    mag_r = None
                    mag_i = None
                else:
                    mag_g = None
                    mag_r = None
                    mag_i = None
            else:
                mag_g = None
                mag_r = None
                mag_i = None

            return {
                'catalog': 'GLADE+',
                'objid': int(best_row['GLADE+']) if 'GLADE+' in best_row.colnames else None,
                'ra': safe_float(best_row['RAJ2000']),
                'dec': safe_float(best_row['DEJ2000']),
                'mag_g': mag_g,
                'mag_r': mag_r,
                'mag_i': mag_i,
                'mag_z': None,
                'redshift': safe_float(best_row['zhelio']) if 'zhelio' in best_row.colnames else None,
                'separation_arcmin': best_sep
            }

        except Exception as e:
            logger.debug("GLADE+ query failed for (%.4f, %.4f): %s", ra, dec, e)
            return None

    @staticmethod
    def classify_morphology(galaxy_info: Dict[str, Any]) -> str:
        """
        Classify galaxy morphology based on color and magnitude properties.

        Uses g-r and r-i colors to distinguish:
        - Red sequence (elliptical): g-r > 0.55 AND r-i > 0.15
        - Blue cloud (spiral): g-r < 0.45 AND r-i < 0.15
        - Green valley (uncertain): intermediate colors

        Args:
            galaxy_info: Dictionary with galaxy properties (magnitudes, etc.)

        Returns:
            Morphology classification: 'elliptical', 'spiral', 'uncertain', 'unknown'
        """
        try:
            if not galaxy_info:
                return 'unknown'

            mag_g = galaxy_info.get('mag_g')
            mag_r = galaxy_info.get('mag_r')
            mag_i = galaxy_info.get('mag_i')

            if mag_g is None or mag_r is None or mag_i is None:
                # If we have a catalog match but no optical colors, return 'uncertain'
                # This is better than 'unknown' since we confirmed a galaxy exists
                catalog = galaxy_info.get('catalog', '')
                redshift = galaxy_info.get('redshift')
                if catalog or redshift is not None:
                    logger.debug("Morphology: uncertain (catalog=%s, z=%s, no optical colors)",
                                catalog, redshift)
                    return 'uncertain'
                logger.debug("Morphology: unknown (no magnitudes or catalog)")
                return 'unknown'

            # Calculate colors
            g_r = mag_g - mag_r
            r_i = mag_r - mag_i

            # Red sequence (early-type) galaxies
            if g_r > 0.55 and r_i > 0.15:
                logger.debug("Morphology: elliptical (g-r=%.2f, r-i=%.2f)", g_r, r_i)
                return 'elliptical'

            # Blue cloud (late-type) galaxies
            elif g_r < 0.45 and r_i < 0.15:
                logger.debug("Morphology: spiral (g-r=%.2f, r-i=%.2f)", g_r, r_i)
                return 'spiral'

            else:
                logger.debug("Morphology: uncertain (g-r=%.2f, r-i=%.2f)", g_r, r_i)
                return 'uncertain'

        except Exception as e:
            logger.warning("Error classifying morphology: %s", e)
            return 'unknown'
