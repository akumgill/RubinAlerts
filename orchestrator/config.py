"""LLAMAS-specific configuration for the MAGNETS observing plan generator."""

import logging
from dataclasses import dataclass, field

from astropy.coordinates import EarthLocation
import astropy.units as u

logger = logging.getLogger(__name__)


@dataclass
class LLAMASConfig:
    """Configuration for the LLAMAS IFU spectrograph at Magellan/Baade."""

    # LCO site
    latitude: float = -29.0146
    longitude: float = -70.6926
    elevation_m: float = 2380

    # LLAMAS instrument
    overhead_minutes: float = 1.0        # IFU advantage — no slit alignment
    # Per-target operations time beyond the IFU overhead: acquisition/guiding
    # setup, plus slew charged from the previous pointing. The 1-min overhead
    # assumption was justified for CLUSTERED DDF targets; wide-sky plans slew
    # across the whole hemisphere, so slew must be modeled (2026-07 review).
    acquisition_buffer_minutes: float = 2.0
    slew_rate_deg_per_min: float = 60.0  # ~1 deg/s incl. settle, coarse model
    # Wall-clock block reserved for each MID-NIGHT standard (2x30s + acquire).
    # Start/end standards live in twilight and consume no dark time.
    std_block_minutes: float = 6.0
    max_airmass: float = 1.6
    gap_fill_max_minutes: float = 5.0

    # Standard star criteria
    std_min_vmag: float = 9.0
    std_max_vmag: float = 12.0
    std_ideal_vmag: float = 10.5
    std_max_airmass: float = 1.5
    # Cadence for mid-night standard insertion. In addition to the start/end
    # standards, insert one standard roughly every this many hours of night
    # so a long (6+ hr) night carries 2-3 standards for spectrophotometric
    # calibration (design spec: interleave=true, observe_per_night=2-3).
    # Nights shorter than this cadence get only the start/end pair.
    standard_interleave_hours: float = 3.5

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

    # ------------------------------------------------------------------
    # Light-curve phase preference (per-program peak vs rising)
    # ------------------------------------------------------------------
    # Gaussian width (days) of the phase weight w_time = exp(-dt²/2τ²). Matches
    # the alert-pipeline merit tau (core.magellan_planning) and
    # normalize.PHASE_WEIGHT_TAU_DAYS, so a program's phase factor is on the
    # same footing as the alert-sourced phase weight.
    phase_tau_days: float = 10.0
    # Preferred time-from-peak (delta_t, in DAYS) per phase preference. 'peak'
    # standardization science wants the SALT epoch (dt=0); 'rising' progenitor/
    # CSM/flash science wants ~7 d before peak (dt=-7). The exact rising offset
    # is a tunable science-policy choice, not a hard instrument constant.
    phase_preference_offsets: dict = field(default_factory=lambda: {
        'peak': 0.0, 'rising': -7.0,
    })
    # Half-width (days) of the 'peak' phase bucket used by the per-target
    # integration ledger (commit 2): |dt| <= window -> peak, dt < -window ->
    # rising, dt > window -> declining.
    phase_bucket_window_days: float = 5.0

    # Sub-exposure splitting. A long integration is broken into N sub-exposures
    # each no longer than ``max_single_exposure_sec`` to limit cosmic-ray
    # accumulation per frame (CR mitigation; frames are later co-added).
    # Per-frame exposure is rounded to a multiple of ``exposure_round_sec`` for
    # clean, schedulable exposure strings.
    max_single_exposure_sec: float = 900.0   # 15 min cap per sub-exposure (CR mitigation)
    exposure_round_sec: float = 10.0         # round per-frame exposure to this granularity

    @property
    def location(self) -> EarthLocation:
        """Astropy EarthLocation for Las Campanas Observatory."""
        return EarthLocation(
            lat=self.latitude * u.deg,
            lon=self.longitude * u.deg,
            height=self.elevation_m * u.m,
        )


LLAMAS_CONFIG = LLAMASConfig()
