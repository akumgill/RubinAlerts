"""Coordinate utilities for alert matching and calculations."""

import logging
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

logger = logging.getLogger(__name__)


class CoordinateUtils:
    """Utilities for coordinate operations."""

    @staticmethod
    def angular_separation(ra1: float, dec1: float,
                          ra2: float, dec2: float) -> float:
        """
        Calculate angular separation between two points.

        Args:
            ra1, dec1: Coordinates of first point (degrees)
            ra2, dec2: Coordinates of second point (degrees)

        Returns:
            Angular separation in arcseconds
        """
        try:
            coord1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
            coord2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
            separation = coord1.separation(coord2)
            return separation.arcsec
        except Exception as e:
            logger.warning(f"Error calculating separation: {e}")
            return float('inf')

    @staticmethod
    def match_coordinates(ra_list1, dec_list1,
                         ra_list2, dec_list2,
                         tolerance_arcsec: float = 2.0):
        """
        Match coordinates between two lists.

        Args:
            ra_list1, dec_list1: First list of coordinates
            ra_list2, dec_list2: Second list of coordinates
            tolerance_arcsec: Matching tolerance in arcseconds

        Returns:
            Indices of matches (idx1, idx2) and separations
        """
        try:
            coords1 = SkyCoord(ra=ra_list1*u.deg, dec=dec_list1*u.deg)
            coords2 = SkyCoord(ra=ra_list2*u.deg, dec=dec_list2*u.deg)

            idx1, idx2, sep2d, _ = coords1.search_around_sky(
                coords2, tolerance_arcsec*u.arcsec
            )

            return idx1, idx2, sep2d.arcsec

        except Exception as e:
            logger.warning(f"Error matching coordinates: {e}")
            return np.array([]), np.array([]), np.array([])

    @staticmethod
    def radec_to_decimal(ra_str: str, dec_str: str):
        """
        Convert RA/Dec strings to decimal degrees.

        Args:
            ra_str: RA in various formats (e.g., "12:30:45.5")
            dec_str: Dec in various formats

        Returns:
            Tuple of (ra_decimal, dec_decimal)
        """
        try:
            coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
            return coord.ra.deg, coord.dec.deg
        except Exception as e:
            logger.warning(f"Error converting coordinates: {e}")
            return None, None
