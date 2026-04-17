"""LLAMAS-specific configuration for the MAGNETS observing plan generator."""

import logging
from dataclasses import dataclass, field

from astropy.coordinates import EarthLocation
import astropy.units as u

logger = logging.getLogger(__name__)


@dataclass
class LLAMASConfig:
    """Configuration for the LLAMAS IFU spectrograph at Magellan/Clay."""

    # LCO site
    latitude: float = -29.0146
    longitude: float = -70.6926
    elevation_m: float = 2380

    # LLAMAS instrument
    overhead_minutes: float = 1.0        # IFU advantage — no slit alignment
    max_airmass: float = 1.6
    gap_fill_max_minutes: float = 5.0

    # Standard star criteria
    std_min_vmag: float = 9.0
    std_max_vmag: float = 12.0
    std_ideal_vmag: float = 10.5
    std_max_airmass: float = 1.5

    # Twilight
    twilight_sun_alt: float = -18.0      # astronomical twilight

    # Exposure table: (max_z, exposure_minutes, moon_constraint)
    # From Stubbs 2026B proposal Table 1
    exposure_table: list = field(default_factory=lambda: [
        (0.20, 35, 'bright'),
        (0.30, 95, 'grey'),
        (0.35, 100, 'grey'),    # midpoint of 45-160
        (0.40, 180, 'dark'),
    ])

    fallback_exposure_minutes: float = 45.0

    @property
    def location(self) -> EarthLocation:
        """Astropy EarthLocation for Las Campanas Observatory."""
        return EarthLocation(
            lat=self.latitude * u.deg,
            lon=self.longitude * u.deg,
            height=self.elevation_m * u.m,
        )


LLAMAS_CONFIG = LLAMASConfig()
