"""Centralized configuration and constants for RubinAlerts pipeline.

This module consolidates magic numbers, thresholds, and configurable parameters
that were previously scattered throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

# =============================================================================
# Photometry Constants
# =============================================================================

# AB magnitude zero point in nanojansky
# AB mag = -2.5 * log10(flux_nJy) + AB_ZP_NJY
AB_ZP_NJY = 31.4

# Band priority for peak fitting (prefer redder bands for SNe Ia)
BAND_PRIORITY = ['i', 'r', 'z', 'g', 'y', 'u']

# Band effective wavelengths (nm) for color calculations
BAND_WAVELENGTHS = {
    'u': 365, 'g': 480, 'r': 620, 'i': 760, 'z': 870, 'y': 1000,
    'zg': 480, 'zr': 640,  # ZTF bands
    'c': 530, 'o': 680,    # ATLAS bands
}


# =============================================================================
# Merit Function Parameters
# =============================================================================

@dataclass
class MeritConfig:
    """Configuration for the target merit/priority function."""

    # Time weight: Gaussian decay from peak
    tau_days: float = 10.0

    # Magnitude weight: optimal for Magellan spectroscopy
    mag_optimal: float = 20.5
    mag_sigma: float = 1.5
    mag_bright_limit: float = 18.0
    mag_faint_limit: float = 23.0

    # Host morphology weights
    host_weights: Dict[str, float] = field(default_factory=lambda: {
        'elliptical': 1.0,
        'spiral': 0.6,
        'irregular': 0.5,
        'uncertain': 0.7,
        'unknown': 0.7,
    })

    # Extinction penalty scale
    extinction_scale: float = 0.15  # exp(-E(B-V) / scale)

    # Multi-broker bonus
    broker_bonus_per_extra: float = 0.1  # 1.0 + 0.1 * (N - 1)


# =============================================================================
# Observability Parameters
# =============================================================================

@dataclass
class ObservatoryConfig:
    """Las Campanas Observatory parameters."""

    name: str = "Las Campanas"
    latitude: float = -29.0146
    longitude: float = -70.6926
    elevation_m: float = 2380
    timezone: str = "America/Santiago"

    # Observability constraints
    max_airmass: float = 2.0
    min_altitude_deg: float = 30.0
    twilight_sun_altitude: float = -12.0  # nautical twilight

    # Typical night duration (hours)
    night_duration_hours: float = 10.0


# =============================================================================
# Broker Query Parameters
# =============================================================================

@dataclass
class BrokerConfig:
    """Configuration for broker queries."""

    # Default query parameters
    default_days_back: int = 30
    default_min_probability: float = 0.3
    default_max_candidates: int = 200

    # Cross-match tolerances (arcseconds)
    coord_match_tolerance: float = 2.0

    # Circuit breaker settings
    max_consecutive_failures: int = 3
    failure_reset_hours: int = 1

    # HTTP timeouts (seconds)
    http_timeout: int = 60
    http_timeout_short: int = 10


# =============================================================================
# Pipeline Thresholds
# =============================================================================

@dataclass
class PipelineConfig:
    """Pipeline processing thresholds."""

    # Minimum points for light curve fitting
    min_lc_points: int = 5
    min_snr_points: int = 5

    # Peak fitting constraints
    min_peak_snr: float = 3.0
    max_delta_t_days: float = 60.0

    # Quality cuts
    min_n_bands: int = 2
    max_fit_chi2: float = 10.0

    # Exposure time estimation (minutes)
    default_exposure_minutes: float = 30.0

    # Report generation
    max_light_curves_per_page: int = 4
    max_targets_in_sequence: int = 20


# =============================================================================
# File Paths
# =============================================================================

@dataclass
class PathConfig:
    """Default paths for data and outputs."""

    cache_dir: str = "./cache/data"
    output_base_dir: str = "./nights"
    log_dir: str = "./logs"

    # Style files
    mpl_style: str = "./utils/rubin.mplstyle"

    def ensure_dirs(self):
        """Create directories if they don't exist."""
        for path in [self.cache_dir, self.output_base_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)


# =============================================================================
# Default Instances
# =============================================================================

# Create default config instances for easy import
MERIT_CONFIG = MeritConfig()
OBSERVATORY_CONFIG = ObservatoryConfig()
BROKER_CONFIG = BrokerConfig()
PIPELINE_CONFIG = PipelineConfig()
PATH_CONFIG = PathConfig()


def get_config() -> Dict:
    """Return all configuration as a dictionary for logging/serialization."""
    return {
        'merit': MERIT_CONFIG.__dict__,
        'observatory': OBSERVATORY_CONFIG.__dict__,
        'broker': BROKER_CONFIG.__dict__,
        'pipeline': PIPELINE_CONFIG.__dict__,
        'paths': PATH_CONFIG.__dict__,
    }
