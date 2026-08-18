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
# Score Function Parameters (PI-approved ranking, 2026-08-18)
# =============================================================================

@dataclass
class ScoreConfig:
    """Configuration for the PI-approved selection score (2026-08-18).

    score = P x V(z) x G x U, computed ALONGSIDE the legacy merit (which stays
    in the outputs for continuity):

      P  — usable-cosmology-Ia probability: w_prob x w_iaspec x w_lcq, where
           w_lcq is a light-curve-quality factor from the SALT2 color error.
      V  — Hubble-diagram sample value: inverse density of the already-observed
           (+ community prior) sample in Delta-z bins.
      G  — information gain from a spectrum: near-zero when the object is
           already spec-typed AND has a spec-z, graded in between.
      U  — urgency: deadline-shaped in REST-FRAME days past peak (replaces the
           Gaussian w_time as the ranking's phase factor).

    EVERY DEFAULT BELOW IS PROVISIONAL pending PI (Chris) review — they encode
    the 2026-08-18 proposal, not a calibrated policy.
    """

    # --- P: light-curve-quality factor w_lcq from SALT2 color error ---
    # w_lcq = clip(1 / (1 + (salt_c_err / c_err_ref)^2), floor, 1.0);
    # missing/non-finite c_err -> neutral 1.0 (codebase convention).
    lcq_c_err_ref: float = 0.06     # provisional
    lcq_floor: float = 0.4          # provisional

    # --- V(z): Hubble-diagram sample-density value ---
    v_bin_width: float = 0.05       # Delta-z bin width over [0, v_z_max]
    v_z_max: float = 0.5
    # prior(bin) = v_prior_amp * exp(-z_mid / v_prior_scale): community low-z
    # saturation (nearby Hubble-diagram bins are already full). Provisional.
    v_prior_amp: float = 30.0
    v_prior_scale: float = 0.08
    v_floor: float = 0.05           # after max-bin normalization to 1.0
    v_unknown_default: float = 0.5  # unknown z when NO night candidate has one
    # Ledger of past observations (orchestrator target ledger) counted into
    # n_eff per bin; missing/unreadable -> counts of 0 (prior only).
    ledger_path: str = 'target_ledger.json'

    # --- G: information gain of taking a spectrum ---
    g_both: float = 0.05            # spec-typed AND spec-z: nearly nothing to gain
    # Chris decision 2026-08-18: an already-typed object without a spec-z is
    # worth LITTLE, not most-of-a-slot — host redshifts come later in batch
    # via MOS, so the missing z barely argues for a live spectrum now
    # (was 0.7 in the original proposal).
    g_type_only: float = 0.15
    g_z_only: float = 0.9           # spec-z but untyped: type is the gain
    g_neither: float = 1.0

    # --- U: urgency, deadline-shaped in rest-frame days past peak ---
    # p = delta_t / (1 + z) (rest-frame; observer-frame when z unknown).
    # U = 1.0 for p <= u_flat_rest_days (including all pre-peak);
    # U = exp(-(p - flat) / u_tau_rest_days) beyond, floored at u_floor.
    u_flat_rest_days: float = 5.0   # provisional
    u_tau_rest_days: float = 12.0   # provisional
    u_floor: float = 0.05           # provisional


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
SCORE_CONFIG = ScoreConfig()
OBSERVATORY_CONFIG = ObservatoryConfig()
BROKER_CONFIG = BrokerConfig()
PIPELINE_CONFIG = PipelineConfig()
PATH_CONFIG = PathConfig()


def get_config() -> Dict:
    """Return all configuration as a dictionary for logging/serialization."""
    return {
        'merit': MERIT_CONFIG.__dict__,
        'score': SCORE_CONFIG.__dict__,
        'observatory': OBSERVATORY_CONFIG.__dict__,
        'broker': BROKER_CONFIG.__dict__,
        'pipeline': PIPELINE_CONFIG.__dict__,
        'paths': PATH_CONFIG.__dict__,
    }
