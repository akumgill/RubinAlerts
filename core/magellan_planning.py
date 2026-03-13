"""Magellan telescope observing plan generation for SN Ia follow-up.

Computes merit scores for spectroscopic follow-up prioritization and
generates target catalogs in the Magellan TCS 16-field format.

Includes:
- Merit function with P(Ia), host morphology, extinction, multi-broker agreement
- Moon position/phase calculations and sky brightness penalties
- Exposure time estimation based on magnitude and conditions
- Optimal observing window computation
- Priority-based scheduling
"""

import logging
from datetime import datetime
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body
from astropy.time import Time
import astropy.units as u

logger = logging.getLogger(__name__)

# Las Campanas Observatory
LAS_CAMPANAS = EarthLocation(
    lat=-29.0146 * u.deg,
    lon=-70.6926 * u.deg,
    height=2516 * u.m,
)

# DDF short names for Magellan object IDs (max ~3 chars)
DDF_SHORT = {
    'COSMOS': 'COS',
    'XMM-LSS': 'XMM',
    'ECDFS': 'ECD',
    'ELAIS-S1': 'ELS',
    'EDFS_a': 'EDa',
    'EDFS_b': 'EDb',
    'M49': 'M49',
}

# Instrument presets for Magellan spectrographs
INSTRUMENT_PRESETS = {
    'LDSS3C': {
        'rotator_angle': 0.0,
        'rotator_mode': 'GRV',
        'description': 'LDSS3C long-slit/MOS',
    },
    'IMACS-f2': {
        'rotator_angle': 0.0,
        'rotator_mode': 'GRV',
        'description': 'IMACS f/2 short camera',
    },
    'IMACS-f4': {
        'rotator_angle': 0.0,
        'rotator_mode': 'GRV',
        'description': 'IMACS f/4 long camera',
    },
    'MagE': {
        'rotator_angle': 0.0,
        'rotator_mode': 'GRV',
        'description': 'MagE echellette',
    },
    'FIRE-echelle': {
        'rotator_angle': 0.0,
        'rotator_mode': 'GRV',
        'description': 'FIRE IR echelle',
    },
    'FIRE-longslit': {
        'rotator_angle': 0.0,
        'rotator_mode': 'GRV',
        'description': 'FIRE IR longslit',
    },
}


# ---------------------------------------------------------------------------
# Merit function
# ---------------------------------------------------------------------------

# Host morphology weights for Ia classification confidence
# Elliptical hosts have essentially zero CC SN rate, so high Ia confidence
# Spiral hosts can have both Ia and CC SNe, so lower confidence
HOST_MORPHOLOGY_WEIGHTS = {
    'elliptical': 1.0,   # High confidence — CC SNe don't occur in E/S0
    'uncertain': 0.8,    # Intermediate — could be either type
    'spiral': 0.6,       # Lower confidence — need spectrum to distinguish
    'unknown': 0.7,      # No host info — use neutral weight
}

# Reference exposure times for SN classification at Magellan (minutes)
# Based on LDSS3C with 1" seeing, S/N~20 for classification
EXPOSURE_TIME_REF = {
    'mag_ref': 20.0,      # Reference magnitude
    'time_ref': 45.0,     # Minutes at reference mag in dark time
    'mag_floor': 17.0,    # Minimum exposure (bright targets)
    'mag_ceiling': 22.5,  # Maximum feasible magnitude
    'time_min': 10.0,     # Minimum exposure minutes
    'time_max': 180.0,    # Maximum practical exposure
}

# Moon impact parameters
MOON_PARAMS = {
    'full_penalty': 0.5,      # Merit multiplier at full moon + close separation
    'separation_critical': 30.0,  # Degrees — strong impact below this
    'separation_safe': 60.0,      # Degrees — minimal impact above this
}


def compute_merit(delta_t, peak_mag,
                  ia_prob=None, host_morphology=None,
                  extinction_ebv=None, num_brokers=None,
                  moon_penalty=None,
                  salt_chi2_dof=None, absolute_mag=None,
                  tau=10.0, mag_optimal=20.5,
                  mag_bright=18.0, mag_faint=23.0):
    """Compute spectroscopic follow-up merit score.

    M = W_t * W_m * W_p * W_h * W_ext * W_broker * W_moon * W_salt * W_absmag

    W_t = exp(-delta_t^2 / (2 * tau^2))        Gaussian time weight
    W_m = exp(-((m - m_opt) / sigma_m)^2)      Gaussian magnitude weight
    W_p = ia_prob                               Classifier probability (0-1)
    W_h = morphology weight                     Host galaxy type weight
    W_ext = exp(-E(B-V) / 0.15)                Extinction penalty
    W_broker = 1.0 + 0.1*(num_brokers - 1)     Multi-broker bonus
    W_moon = moon_penalty                       Moon proximity penalty (0-1)
    W_salt = SALT2 chi2/dof quality            Good template fit bonus (0.5-1.2)
    W_absmag = absolute mag consistency        SN Ia M_B ~ -19.3 (0.5-1.0)

    Parameters
    ----------
    delta_t : float or array-like
        Time since peak in days (MJD_now - peak_mjd). Sign preserved but
        the Gaussian uses delta_t^2, so it is symmetric.
    peak_mag : float or array-like
        Estimated peak apparent magnitude (AB). Preferably extinction-
        corrected (peak_mag_corrected) — use that column when available.
    ia_prob : float or array-like, optional
        Type Ia classification probability from ML classifier (0-1).
        If None, this factor is omitted (equivalent to 1.0).
    host_morphology : str or array-like, optional
        Host galaxy morphology: 'elliptical', 'spiral', 'uncertain', 'unknown'.
        If None, this factor is omitted (equivalent to 1.0).
    extinction_ebv : float or array-like, optional
        Galactic E(B-V) extinction. Lower is better.
        If None, this factor is omitted (equivalent to 1.0).
    num_brokers : int or array-like, optional
        Number of brokers that detected this object. More = higher confidence.
        If None, this factor is omitted (equivalent to 1.0).
    moon_penalty : float or array-like, optional
        Moon proximity/phase penalty factor (0-1). 1.0 = no penalty.
        If None, this factor is omitted (equivalent to 1.0).
    salt_chi2_dof : float or array-like, optional
        SALT2/SALT3 fit chi2/dof. Lower is better (good fit to SN Ia template).
        If None, this factor is omitted (equivalent to 1.0).
    absolute_mag : float or array-like, optional
        Absolute magnitude (peak_mag - distmod). SNe Ia have M_B ~ -19.3.
        If None, this factor is omitted (equivalent to 1.0).
    tau : float
        Time decay half-width in days (default 10).
    mag_optimal : float
        Optimal magnitude for the telescope/instrument (default 20.5).
    mag_bright, mag_faint : float
        Magnitude range bounds. sigma_m = (mag_faint - mag_bright) / 4.

    Returns
    -------
    float or array-like
        Merit score in [0, 1]. NaN if inputs are NaN.
    """
    delta_t = np.asarray(delta_t, dtype=float)
    peak_mag = np.asarray(peak_mag, dtype=float)

    sigma_m = (mag_faint - mag_bright) / 4.0

    w_t = np.exp(-delta_t**2 / (2.0 * tau**2))
    w_m = np.exp(-((peak_mag - mag_optimal) / sigma_m)**2)

    merit = w_t * w_m

    # Classifier probability weight
    if ia_prob is not None:
        ia_prob = np.asarray(ia_prob, dtype=float)
        # Clamp to [0.1, 1.0] to avoid zeroing out candidates entirely
        w_p = np.clip(ia_prob, 0.1, 1.0)
        merit = merit * w_p

    # Host morphology weight
    if host_morphology is not None:
        if isinstance(host_morphology, str):
            w_h = HOST_MORPHOLOGY_WEIGHTS.get(host_morphology, 0.7)
        else:
            # Array of morphology strings
            w_h = np.array([HOST_MORPHOLOGY_WEIGHTS.get(m, 0.7)
                           for m in host_morphology])
        merit = merit * w_h

    # Galactic extinction penalty — lower E(B-V) is better
    if extinction_ebv is not None:
        extinction_ebv = np.asarray(extinction_ebv, dtype=float)
        # exp(-E(B-V)/0.15): E(B-V)=0 → 1.0, E(B-V)=0.15 → 0.37, E(B-V)=0.3 → 0.14
        w_ext = np.exp(-np.clip(extinction_ebv, 0, 1.0) / 0.15)
        w_ext = np.where(np.isfinite(extinction_ebv), w_ext, 1.0)
        merit = merit * w_ext

    # Multi-broker agreement bonus
    if num_brokers is not None:
        num_brokers = np.asarray(num_brokers, dtype=float)
        # 1 broker = 1.0, 2 brokers = 1.1, 3 brokers = 1.2
        w_broker = 1.0 + 0.1 * np.clip(num_brokers - 1, 0, 3)
        w_broker = np.where(np.isfinite(num_brokers), w_broker, 1.0)
        merit = merit * w_broker

    # Moon penalty (pre-computed from phase and separation)
    if moon_penalty is not None:
        moon_penalty = np.asarray(moon_penalty, dtype=float)
        # Clamp to [0.3, 1.0] — don't zero out, just penalize
        w_moon = np.clip(moon_penalty, 0.3, 1.0)
        w_moon = np.where(np.isfinite(moon_penalty), w_moon, 1.0)
        merit = merit * w_moon

    # SALT2 chi2/dof quality — good template fit = higher confidence it's a SN Ia
    if salt_chi2_dof is not None:
        salt_chi2_dof = np.asarray(salt_chi2_dof, dtype=float)
        # chi2/dof < 1.5 is excellent (bonus), 1.5-3 is acceptable (neutral),
        # > 3 is poor fit (penalty). Use sigmoid-like mapping.
        # w_salt = 1.2 for chi2/dof=1, 1.0 for chi2/dof=2, 0.5 for chi2/dof>5
        w_salt = 1.2 - 0.7 * (1 - np.exp(-np.maximum(salt_chi2_dof - 1, 0) / 2))
        w_salt = np.clip(w_salt, 0.5, 1.2)
        w_salt = np.where(np.isfinite(salt_chi2_dof), w_salt, 1.0)
        merit = merit * w_salt

    # Absolute magnitude consistency with SN Ia (M_B ~ -19.3 ± 0.5)
    if absolute_mag is not None:
        absolute_mag = np.asarray(absolute_mag, dtype=float)
        # SNe Ia have M_B ~ -19.3. Gaussian penalty for deviations.
        # sigma = 0.7 mag to allow for intrinsic scatter + stretch/color variations
        M_Ia = -19.3
        sigma_M = 0.7
        w_absmag = np.exp(-((absolute_mag - M_Ia) / sigma_M)**2 / 2)
        # Clamp to [0.3, 1.0] to not completely reject outliers
        w_absmag = np.clip(w_absmag, 0.3, 1.0)
        w_absmag = np.where(np.isfinite(absolute_mag), w_absmag, 1.0)
        merit = merit * w_absmag

    # NaN propagation is automatic from numpy, but be explicit
    merit = np.where(np.isfinite(delta_t) & np.isfinite(peak_mag),
                     merit, np.nan)
    return merit


def compute_merit_breakdown(delta_t, peak_mag,
                            ia_prob=None, host_morphology=None,
                            extinction_ebv=None, num_brokers=None,
                            moon_penalty=None,
                            salt_chi2_dof=None, absolute_mag=None,
                            tau=10.0, mag_optimal=20.5,
                            mag_bright=18.0, mag_faint=23.0):
    """Compute merit score and return individual component weights.

    Same calculation as compute_merit(), but returns a dict with all factors.

    Returns
    -------
    dict with keys:
        'merit': Final combined merit score
        'w_time': Time weight (Gaussian decay from peak)
        'w_mag': Magnitude weight (Gaussian around optimal)
        'w_prob': Classifier probability weight
        'w_host': Host morphology weight
        'w_ext': Extinction penalty
        'w_broker': Multi-broker bonus
        'w_moon': Moon penalty
        'w_salt': SALT2 fit quality weight
        'w_absmag': Absolute magnitude consistency weight
    """
    delta_t = np.asarray(delta_t, dtype=float)
    peak_mag = np.asarray(peak_mag, dtype=float)

    sigma_m = (mag_faint - mag_bright) / 4.0

    w_time = np.exp(-delta_t**2 / (2.0 * tau**2))
    w_mag = np.exp(-((peak_mag - mag_optimal) / sigma_m)**2)

    # Initialize all weights to 1.0
    w_prob = np.ones_like(delta_t)
    w_host = np.ones_like(delta_t)
    w_ext = np.ones_like(delta_t)
    w_broker = np.ones_like(delta_t)
    w_moon = np.ones_like(delta_t)
    w_salt = np.ones_like(delta_t)
    w_absmag = np.ones_like(delta_t)

    if ia_prob is not None:
        ia_prob = np.asarray(ia_prob, dtype=float)
        w_prob = np.clip(ia_prob, 0.1, 1.0)
        w_prob = np.where(np.isfinite(ia_prob), w_prob, 1.0)

    if host_morphology is not None:
        if isinstance(host_morphology, str):
            w_host = np.full_like(delta_t, HOST_MORPHOLOGY_WEIGHTS.get(host_morphology, 0.7))
        elif np.ndim(host_morphology) == 0:
            # Scalar non-string (e.g., nan or single value)
            w_host = np.full_like(delta_t, HOST_MORPHOLOGY_WEIGHTS.get(str(host_morphology), 0.7))
        else:
            # Array of morphology strings
            w_host = np.array([HOST_MORPHOLOGY_WEIGHTS.get(str(m), 0.7)
                              for m in host_morphology], dtype=float)

    if extinction_ebv is not None:
        extinction_ebv = np.asarray(extinction_ebv, dtype=float)
        w_ext = np.exp(-np.clip(extinction_ebv, 0, 1.0) / 0.15)
        w_ext = np.where(np.isfinite(extinction_ebv), w_ext, 1.0)

    if num_brokers is not None:
        num_brokers = np.asarray(num_brokers, dtype=float)
        w_broker = 1.0 + 0.1 * np.clip(num_brokers - 1, 0, 3)
        w_broker = np.where(np.isfinite(num_brokers), w_broker, 1.0)

    if moon_penalty is not None:
        moon_penalty = np.asarray(moon_penalty, dtype=float)
        w_moon = np.clip(moon_penalty, 0.3, 1.0)
        w_moon = np.where(np.isfinite(moon_penalty), w_moon, 1.0)

    if salt_chi2_dof is not None:
        salt_chi2_dof = np.asarray(salt_chi2_dof, dtype=float)
        # chi2/dof < 1.5 is excellent (bonus), 1.5-3 is acceptable (neutral),
        # > 3 is poor fit (penalty)
        w_salt = 1.2 - 0.7 * (1 - np.exp(-np.maximum(salt_chi2_dof - 1, 0) / 2))
        w_salt = np.clip(w_salt, 0.5, 1.2)
        w_salt = np.where(np.isfinite(salt_chi2_dof), w_salt, 1.0)

    if absolute_mag is not None:
        absolute_mag = np.asarray(absolute_mag, dtype=float)
        # SNe Ia have M_B ~ -19.3. Gaussian penalty for deviations.
        M_Ia = -19.3
        sigma_M = 0.7
        w_absmag = np.exp(-((absolute_mag - M_Ia) / sigma_M)**2 / 2)
        w_absmag = np.clip(w_absmag, 0.3, 1.0)
        w_absmag = np.where(np.isfinite(absolute_mag), w_absmag, 1.0)

    merit = w_time * w_mag * w_prob * w_host * w_ext * w_broker * w_moon * w_salt * w_absmag
    merit = np.where(np.isfinite(delta_t) & np.isfinite(peak_mag), merit, np.nan)

    return {
        'merit': merit,
        'w_time': w_time,
        'w_mag': w_mag,
        'w_prob': w_prob,
        'w_host': w_host,
        'w_ext': w_ext,
        'w_broker': w_broker,
        'w_moon': w_moon,
        'w_salt': w_salt,
        'w_absmag': w_absmag,
    }


# ---------------------------------------------------------------------------
# Moon calculations
# ---------------------------------------------------------------------------

def get_moon_info(obs_time, site=LAS_CAMPANAS):
    """Get moon position and illumination at a given time.

    Parameters
    ----------
    obs_time : astropy.time.Time
        Observation time (or array of times).
    site : EarthLocation
        Observatory location.

    Returns
    -------
    dict with keys:
        'moon_altaz': AltAz frame with moon position
        'moon_ra', 'moon_dec': Moon coordinates (degrees)
        'moon_alt': Moon altitude (degrees)
        'illumination': Moon illumination fraction (0-1)
        'phase_angle': Moon phase angle (degrees, 0=full, 180=new)
    """
    # Get moon position
    moon = get_body('moon', obs_time)
    sun = get_sun(obs_time)

    # Transform to AltAz
    frame = AltAz(obstime=obs_time, location=site)
    moon_altaz = moon.transform_to(frame)

    # Moon illumination (approximate)
    # Phase angle: angle Sun-Moon-Earth
    elongation = sun.separation(moon)
    phase_angle = np.arctan2(
        sun.distance * np.sin(elongation),
        moon.distance - sun.distance * np.cos(elongation)
    )
    illumination = (1 + np.cos(phase_angle)) / 2

    return {
        'moon_coord': moon,
        'moon_ra': moon.ra.deg,
        'moon_dec': moon.dec.deg,
        'moon_alt': moon_altaz.alt.deg,
        'illumination': float(illumination) if np.ndim(illumination) == 0 else illumination,
        'phase_angle': np.degrees(phase_angle),
    }


def compute_moon_separation(target_ra, target_dec, moon_ra, moon_dec):
    """Compute angular separation between target(s) and moon.

    Parameters
    ----------
    target_ra, target_dec : float or array-like
        Target coordinates (degrees).
    moon_ra, moon_dec : float
        Moon coordinates (degrees).

    Returns
    -------
    float or array-like
        Angular separation in degrees.
    """
    target = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
    moon = SkyCoord(ra=moon_ra * u.deg, dec=moon_dec * u.deg)
    return target.separation(moon).deg


def compute_moon_penalty(separation_deg, illumination, moon_alt=None):
    """Compute merit penalty factor based on moon proximity and phase.

    Parameters
    ----------
    separation_deg : float or array-like
        Angular separation from moon (degrees).
    illumination : float
        Moon illumination fraction (0-1, where 1 = full moon).
    moon_alt : float, optional
        Moon altitude (degrees). If below horizon, no penalty.

    Returns
    -------
    float or array-like
        Penalty factor (0-1). 1.0 = no penalty, lower = worse.
    """
    separation_deg = np.asarray(separation_deg, dtype=float)

    # If moon is below horizon, no penalty
    if moon_alt is not None and moon_alt < 0:
        return np.ones_like(separation_deg)

    # Separation factor: 1.0 if > safe, decreasing to 0.3 if < critical
    sep_safe = MOON_PARAMS['separation_safe']
    sep_crit = MOON_PARAMS['separation_critical']

    sep_factor = np.clip(
        (separation_deg - sep_crit) / (sep_safe - sep_crit),
        0.0, 1.0
    )
    # Map to [0.3, 1.0] range
    sep_factor = 0.3 + 0.7 * sep_factor

    # Illumination factor: full moon = stronger penalty
    # New moon (illum~0) = no penalty, full moon (illum~1) = full penalty
    illum_factor = 1.0 - 0.5 * illumination  # ranges from 1.0 to 0.5

    # Combined penalty
    penalty = sep_factor * illum_factor

    return penalty


def estimate_exposure_time(peak_mag, moon_illumination=0.0, airmass=1.2):
    """Estimate spectroscopic exposure time for SN classification.

    Based on LDSS3C at Magellan with 1" seeing, targeting S/N~20.

    Parameters
    ----------
    peak_mag : float or array-like
        Target magnitude (AB).
    moon_illumination : float
        Moon illumination fraction (0-1). Increases required time.
    airmass : float or array-like
        Airmass at observation. Increases required time.

    Returns
    -------
    float or array-like
        Estimated exposure time in minutes.
    """
    peak_mag = np.asarray(peak_mag, dtype=float)
    airmass = np.asarray(airmass, dtype=float)

    ref = EXPOSURE_TIME_REF

    # Base time scales as 10^(0.4 * (mag - mag_ref)) for background-limited
    mag_diff = np.clip(peak_mag, ref['mag_floor'], ref['mag_ceiling']) - ref['mag_ref']
    base_time = ref['time_ref'] * np.power(10, 0.4 * mag_diff)

    # Moon penalty: up to 2x longer for bright moon
    moon_factor = 1.0 + moon_illumination

    # Airmass penalty: ~20% longer per 0.5 airmass above 1.0
    airmass_factor = 1.0 + 0.4 * np.clip(airmass - 1.0, 0, 2.0)

    total_time = base_time * moon_factor * airmass_factor

    # Clamp to practical range
    total_time = np.clip(total_time, ref['time_min'], ref['time_max'])

    return total_time


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

def _get_twilight_times(obs_date, site=LAS_CAMPANAS, sun_alt_limit=-12.0):
    """Compute evening and morning twilight for a given UT date.

    Parameters
    ----------
    obs_date : str or astropy.time.Time
        UT date of the observing night (e.g., '2026-03-15').
    site : EarthLocation
        Observatory location.
    sun_alt_limit : float
        Sun altitude threshold in degrees (default -12 = nautical twilight).

    Returns
    -------
    (evening_twilight, morning_twilight) as astropy.time.Time objects,
    or (None, None) if Sun never drops below threshold.
    """
    if isinstance(obs_date, str):
        obs_date = Time(obs_date)

    # Search from local noon to next local noon (UT)
    # Las Campanas is ~UT-5, so local noon is ~17:00 UT
    t_start = Time(obs_date.iso[:10] + 'T17:00:00', format='isot')
    t_end = t_start + 1.0 * u.day

    times = t_start + np.linspace(0, 1, 289) * u.day  # 5-min grid over 24h
    frame = AltAz(obstime=times, location=site)
    sun_alts = get_sun(times).transform_to(frame).alt.deg

    # Find transitions below/above the threshold
    below = sun_alts < sun_alt_limit

    if not below.any():
        return None, None  # Sun never sets low enough (polar day)
    if below.all():
        return times[0], times[-1]  # Sun always below (polar night)

    # Evening twilight: first time Sun drops below threshold
    transitions_down = np.where(np.diff(below.astype(int)) == 1)[0]
    transitions_up = np.where(np.diff(below.astype(int)) == -1)[0]

    evening = times[transitions_down[0] + 1] if len(transitions_down) > 0 else times[0]
    morning = times[transitions_up[-1]] if len(transitions_up) > 0 else times[-1]

    return evening, morning


def filter_observable_targets(targets_df, obs_date,
                              site=LAS_CAMPANAS,
                              max_airmass=2.0, min_hours_up=1.0,
                              include_moon=True, include_exposure=True):
    """Filter targets observable from Las Campanas on a given night.

    Parameters
    ----------
    targets_df : pd.DataFrame
        Must have 'ra' and 'dec' columns (degrees).
        Optional: 'peak_mag' for exposure time estimation.
    obs_date : str
        UT date of observing night (e.g., '2026-03-15').
    site : EarthLocation
        Observatory location.
    max_airmass : float
        Maximum acceptable airmass (default 2.0 = 30 deg altitude).
    min_hours_up : float
        Minimum hours target must be above the airmass limit.
    include_moon : bool
        If True, compute moon separation and penalty.
    include_exposure : bool
        If True, estimate exposure times.

    Returns
    -------
    pd.DataFrame — copy of input filtered to observable targets, with
    added columns:
        transit_alt, min_airmass, hours_observable,
        optimal_time_ut, optimal_airmass,
        moon_separation, moon_penalty, moon_illumination,
        exposure_minutes
    """
    evening, morning = _get_twilight_times(obs_date, site=site)
    if evening is None or morning is None:
        logger.warning("Could not compute twilight for %s", obs_date)
        return targets_df.iloc[0:0].copy()

    dark_hours = (morning - evening).to(u.hour).value
    logger.info("Observing window: %s to %s (%.1f hours dark)",
                evening.iso, morning.iso, dark_hours)

    # Time grid at 15-minute intervals during dark time
    n_steps = max(int(dark_hours * 4), 2)
    times = evening + np.linspace(0, 1, n_steps) * (morning - evening)

    min_alt = np.degrees(np.arcsin(1.0 / max_airmass))  # ~30 deg for airmass 2

    coords = SkyCoord(
        ra=targets_df['ra'].values * u.deg,
        dec=targets_df['dec'].values * u.deg,
    )

    n_targets = len(targets_df)

    # Track altitude for each target at each time step
    all_alts = np.zeros((n_targets, n_steps))
    all_airmass = np.full((n_targets, n_steps), np.inf)

    for i, t in enumerate(times):
        frame = AltAz(obstime=t, location=site)
        alts = coords.transform_to(frame).alt.deg
        all_alts[:, i] = alts
        # Airmass (avoid div by zero for negative altitudes)
        all_airmass[:, i] = np.where(
            alts > 5,
            1.0 / np.sin(np.radians(np.clip(alts, 5, 90))),
            np.inf
        )

    # Compute summary stats
    transit_alts = np.max(all_alts, axis=1)
    hours_up = np.sum(all_alts >= min_alt, axis=1) * (dark_hours / n_steps)

    # Find optimal time (minimum airmass) for each target
    best_idx = np.argmin(all_airmass, axis=1)
    optimal_airmass = all_airmass[np.arange(n_targets), best_idx]
    optimal_times = times[best_idx]

    out = targets_df.copy()
    out['transit_alt'] = transit_alts
    out['min_airmass'] = np.where(
        transit_alts > 0,
        1.0 / np.sin(np.radians(transit_alts)),
        np.inf,
    )
    out['hours_observable'] = hours_up
    out['optimal_time_ut'] = [t.iso[11:16] for t in optimal_times]  # HH:MM
    out['optimal_airmass'] = optimal_airmass

    # Moon calculations
    if include_moon:
        # Get moon position at middle of night
        mid_night = evening + 0.5 * (morning - evening)
        moon_info = get_moon_info(mid_night, site=site)

        moon_sep = compute_moon_separation(
            targets_df['ra'].values,
            targets_df['dec'].values,
            moon_info['moon_ra'],
            moon_info['moon_dec']
        )
        moon_illum = moon_info['illumination']
        moon_alt = moon_info['moon_alt']

        moon_pen = compute_moon_penalty(moon_sep, moon_illum, moon_alt)

        out['moon_separation'] = moon_sep
        out['moon_illumination'] = moon_illum
        out['moon_penalty'] = moon_pen

        # Log moon status
        phase_name = _moon_phase_name(moon_illum)
        logger.info("Moon: %s (%.0f%% illuminated), alt=%.0f°",
                    phase_name, moon_illum * 100, moon_alt)
    else:
        out['moon_separation'] = np.nan
        out['moon_illumination'] = np.nan
        out['moon_penalty'] = 1.0

    # Exposure time estimates
    if include_exposure and 'peak_mag' in targets_df.columns:
        moon_illum = out['moon_illumination'].iloc[0] if include_moon else 0.0
        out['exposure_minutes'] = estimate_exposure_time(
            targets_df['peak_mag'].values,
            moon_illumination=moon_illum,
            airmass=optimal_airmass
        )
    else:
        out['exposure_minutes'] = np.nan

    # Filter by observability
    mask = (hours_up >= min_hours_up)
    n_before = len(out)
    out = out[mask].copy()
    logger.info("Observability filter: %d/%d targets pass (airmass < %.1f, "
                ">= %.1f hours)", len(out), n_before, max_airmass, min_hours_up)

    return out


def _moon_phase_name(illumination):
    """Convert moon illumination to human-readable phase name."""
    if illumination < 0.05:
        return "New"
    elif illumination < 0.35:
        return "Crescent"
    elif illumination < 0.65:
        return "Quarter"
    elif illumination < 0.95:
        return "Gibbous"
    else:
        return "Full"


def prioritize_targets(targets_df, time_critical_days=5.0):
    """Sort targets by observing priority.

    Priority factors:
    1. Time-critical targets (near peak) get highest priority
    2. Within time-critical: sort by merit
    3. Setting targets (optimal time early in night) get priority
    4. Remaining targets sorted by merit

    Parameters
    ----------
    targets_df : pd.DataFrame
        Must have 'merit', 'delta_t', 'optimal_time_ut' columns.
    time_critical_days : float
        Targets within this many days of peak are time-critical.

    Returns
    -------
    pd.DataFrame — sorted by priority (highest first).
    """
    df = targets_df.copy()

    # Priority score components
    # 1. Time-critical bonus (within N days of peak)
    if 'delta_t' in df.columns:
        df['_time_critical'] = np.abs(df['delta_t']) <= time_critical_days
    else:
        df['_time_critical'] = False

    # 2. Setting soon bonus (optimal time in first half of night)
    if 'optimal_time_ut' in df.columns:
        # Parse HH:MM and check if before midnight-ish
        def is_early(time_str):
            if pd.isna(time_str):
                return False
            try:
                hh = int(time_str.split(':')[0])
                # Before 02:00 UT is early evening at Las Campanas
                return hh < 2 or hh >= 22
            except (ValueError, IndexError):
                return False
        df['_setting_soon'] = df['optimal_time_ut'].apply(is_early)
    else:
        df['_setting_soon'] = False

    # 3. Base merit
    df['_merit'] = df.get('merit', 0.0).fillna(0.0)

    # Composite priority score
    df['priority_score'] = (
        df['_time_critical'].astype(float) * 1000 +  # Time-critical first
        df['_setting_soon'].astype(float) * 100 +    # Setting targets next
        df['_merit'] * 10                             # Then by merit
    )

    # Count before dropping temp columns
    n_time_crit = df['_time_critical'].sum()
    n_setting = df['_setting_soon'].sum()

    # Sort by priority (descending)
    df = df.sort_values('priority_score', ascending=False)

    # Clean up temp columns
    df = df.drop(columns=['_time_critical', '_setting_soon', '_merit', 'priority_score'])

    logger.info("Prioritized %d targets (time-critical: %d, setting soon: %d)",
                len(df), n_time_crit, n_setting)

    return df


def optimize_observing_sequence(targets_df, obs_date, max_targets=None,
                                 slew_weight=0.5, merit_weight=0.5,
                                 exposure_minutes=30):
    """Compute optimal observing sequence for a single night.

    Uses a greedy nearest-neighbor approach weighted by:
    - Angular distance (minimize slew time)
    - Merit score (prioritize high-value targets)
    - Visibility windows (observe targets when they're optimal)

    Parameters
    ----------
    targets_df : pd.DataFrame
        Must have: ra, dec, merit, optimal_time_ut.
        Assumes targets are already filtered for observability.
    obs_date : str
        Observing date (e.g., '2026-03-15').
    max_targets : int, optional
        Maximum number of targets to schedule.
    slew_weight : float
        Weight for slew distance penalty (0-1).
    merit_weight : float
        Weight for merit preference (0-1).
    exposure_minutes : float
        Assumed exposure time per target.

    Returns
    -------
    pd.DataFrame with observation sequence, including:
        - obs_order: 1-indexed observation order
        - obs_time_ut: Scheduled observation time (HH:MM)
        - slew_deg: Angular distance from previous target
        - cumulative_time_hr: Running total observing time
    """
    if len(targets_df) == 0:
        return targets_df.copy()

    df = targets_df.copy()

    # Parse optimal_time_ut to float hours for sorting
    def parse_ut_hours(time_str):
        if pd.isna(time_str):
            return 24.0  # Default to end of night
        try:
            parts = str(time_str).split(':')
            hh = int(parts[0])
            mm = int(parts[1]) if len(parts) > 1 else 0
            # Handle times that cross midnight (00-10 UT is late night)
            if hh < 12:
                hh += 24  # Push 00:00-11:59 UT to after 24:00
            return hh + mm / 60
        except (ValueError, IndexError):
            return 24.0

    df['_opt_hours'] = df['optimal_time_ut'].apply(parse_ut_hours)

    # Normalize merit for weighting (0-1 scale)
    merit_vals = df['merit'].fillna(0).values
    if merit_vals.max() > 0:
        merit_norm = merit_vals / merit_vals.max()
    else:
        merit_norm = np.ones(len(df))

    # Greedy nearest-neighbor with merit weighting
    n = len(df)
    if max_targets:
        n = min(n, max_targets)

    visited = [False] * len(df)
    sequence = []
    slews = []

    # Start with the target that's optimal earliest in the night
    current_idx = df['_opt_hours'].idxmin()
    visited[df.index.get_loc(current_idx)] = True
    sequence.append(current_idx)
    slews.append(0.0)

    current_ra = df.loc[current_idx, 'ra']
    current_dec = df.loc[current_idx, 'dec']

    for _ in range(n - 1):
        best_idx = None
        best_score = -np.inf

        for idx in df.index:
            loc = df.index.get_loc(idx)
            if visited[loc]:
                continue

            # Angular distance (approximate, good enough for scheduling)
            d_ra = (df.loc[idx, 'ra'] - current_ra) * np.cos(np.radians(current_dec))
            d_dec = df.loc[idx, 'dec'] - current_dec
            slew_dist = np.sqrt(d_ra**2 + d_dec**2)

            # Score: prefer nearby targets with high merit
            # Normalize slew (assume max useful slew ~60 deg)
            slew_penalty = slew_dist / 60.0
            merit_bonus = merit_norm[loc]

            score = merit_weight * merit_bonus - slew_weight * slew_penalty

            if score > best_score:
                best_score = score
                best_idx = idx
                best_slew = slew_dist

        if best_idx is None:
            break

        visited[df.index.get_loc(best_idx)] = True
        sequence.append(best_idx)
        slews.append(best_slew)

        current_ra = df.loc[best_idx, 'ra']
        current_dec = df.loc[best_idx, 'dec']

    # Build result DataFrame in sequence order
    result = df.loc[sequence].copy()
    result['obs_order'] = range(1, len(sequence) + 1)
    result['slew_deg'] = slews

    # Calculate observation times
    # Night at Las Campanas: ~23:00 UT to ~10:00 UT (next day)
    start_ut = 23.0  # 23:00 UT start
    obs_times = []
    cumulative = 0.0

    for i in range(len(result)):
        current_ut = start_ut + cumulative
        if current_ut >= 24:
            current_ut -= 24  # Wrap to next day
        hh = int(current_ut)
        mm = int((current_ut - hh) * 60)
        obs_times.append(f'{hh:02d}:{mm:02d}')
        cumulative += exposure_minutes / 60.0

    result['obs_time_ut'] = obs_times
    result['cumulative_time_hr'] = np.cumsum([exposure_minutes / 60.0] * len(result))

    # Total slew distance
    total_slew = sum(slews)
    total_hours = len(result) * exposure_minutes / 60.0

    logger.info("Optimized sequence: %d targets, %.1f deg total slew, %.1f hours",
                len(result), total_slew, total_hours)

    # Clean up temp column
    result = result.drop(columns=['_opt_hours'], errors='ignore')

    return result


# ---------------------------------------------------------------------------
# Coordinate formatting
# ---------------------------------------------------------------------------

def radec_to_sexagesimal(ra_deg, dec_deg):
    """Convert RA/Dec (degrees) to Magellan catalog sexagesimal format.

    Returns
    -------
    (ra_str, dec_str) : tuple of str
        RA as 'hh:mm:ss.s', Dec as '+dd:mm:ss' or '-dd:mm:ss'.
    """
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)

    ra_hms = coord.ra.hms
    ra_str = '{:02d}:{:02d}:{:04.1f}'.format(
        int(ra_hms.h), int(abs(ra_hms.m)), abs(ra_hms.s))

    dec_dms = coord.dec.dms
    dec_sign = '+' if dec_dms.d >= 0 else '-'
    dec_str = '{}{:02d}:{:02d}:{:02d}'.format(
        dec_sign, int(abs(dec_dms.d)), int(abs(dec_dms.m)), int(abs(dec_dms.s)))

    return ra_str, dec_str


def sanitize_object_name(raw_name, ddf_field='', max_len=20):
    """Convert an object ID to a Magellan-safe name.

    Allowed characters: alphanumeric, plus, minus, period, underscore.
    Prepends DDF short prefix if available.
    """
    prefix = DDF_SHORT.get(ddf_field, '')

    name = str(raw_name)
    # Keep only allowed characters
    cleaned = ''.join(c if c.isalnum() or c in '+-._' else '_' for c in name)

    if prefix:
        full = prefix + '_' + cleaned
    else:
        full = cleaned

    return full[:max_len]


# ---------------------------------------------------------------------------
# Catalog writer
# ---------------------------------------------------------------------------

def write_magellan_catalog(targets_df, output_path,
                           instrument='LDSS3C',
                           rotator_angle=None,
                           rotator_mode=None,
                           obs_date=None,
                           merit_params=None):
    """Write a Magellan-format target catalog.

    Targets are sorted by RA for efficient telescope slewing.

    Parameters
    ----------
    targets_df : pd.DataFrame
        Must have: ra, dec, object_id (or object_id_alerce_lsst/antares),
        ddf_field. Optional: merit, peak_mag, delta_t, peak_fit_status.
    output_path : str
        Output file path (e.g., 'magellan_targets_20260315.cat').
    instrument : str
        Instrument name (key into INSTRUMENT_PRESETS).
    rotator_angle, rotator_mode : float, str
        Override preset values if provided.
    obs_date : str
        Observing date for header (e.g., '2026-03-15').
    merit_params : dict
        Merit function parameters for header documentation.
    """
    preset = INSTRUMENT_PRESETS.get(instrument, INSTRUMENT_PRESETS['LDSS3C'])
    rot_angle = rotator_angle if rotator_angle is not None else preset['rotator_angle']
    rot_mode = rotator_mode if rotator_mode is not None else preset['rotator_mode']

    # Sort by RA
    df = targets_df.sort_values('ra').reset_index(drop=True)

    lines = []

    # Comment header
    lines.append('# Magellan Observing Catalog — RubinAlerts SN Ia Follow-Up')
    lines.append(f'# Generated: {datetime.utcnow().isoformat()} UTC')
    if obs_date:
        lines.append(f'# Observing night: {obs_date} UT')
    lines.append(f'# Instrument: {instrument} ({preset["description"]})')
    lines.append(f'# Rotator: angle={rot_angle:.1f}, mode={rot_mode}')
    if merit_params:
        lines.append(f'# Merit params: tau={merit_params.get("tau", "?")}d, '
                     f'm_opt={merit_params.get("mag_optimal", "?")}, '
                     f'sigma_m={merit_params.get("sigma_m", "?")}')
    lines.append(f'# Targets: {len(df)} (sorted by RA ascending)')
    lines.append('#')
    lines.append('# ref  name                  RA          Dec       '
                 '  merit  peak_mag  delta_t  DDF        status')
    lines.append('# ---  ----                  --          ---       '
                 '  -----  --------  -------  ---        ------')

    # Per-target comment + data line
    for i, (_, row) in enumerate(df.iterrows()):
        ref = i + 1

        # Resolve object name
        raw_id = row.get('object_id')
        if pd.isna(raw_id):
            raw_id = next(
                (v for col in ('object_id_alerce_lsst', 'object_id_antares',
                               'object_id_ALeRCE-LSST', 'object_id_ANTARES')
                 for v in [row.get(col)] if pd.notna(v)),
                f'target_{ref}'
            )
        ddf = row.get('ddf_field', '')
        name = sanitize_object_name(raw_id, ddf_field=ddf if pd.notna(ddf) else '')

        # Coordinates
        ra_str, dec_str = radec_to_sexagesimal(row['ra'], row['dec'])

        # Diagnostic comment
        merit_val = row.get('merit', np.nan)
        pmag = row.get('peak_mag', np.nan)
        dt = row.get('delta_t', np.nan)
        status = row.get('peak_fit_status', '?')
        ddf_name = ddf if pd.notna(ddf) else '?'

        merit_s = f'{merit_val:.3f}' if pd.notna(merit_val) else '  --  '
        pmag_s = f'{pmag:.2f}' if pd.notna(pmag) else '  --  '
        dt_s = f'{dt:+.1f}d' if pd.notna(dt) else '  --  '

        lines.append(f'# {ref:3d}  {name:20s}  {ra_str:11s} {dec_str:10s}'
                     f'  {merit_s:>6s}  {pmag_s:>8s}  {dt_s:>7s}  '
                     f'{ddf_name:10s} {status}')

        # Data line: 16 fields
        lines.append(
            f'  {ref} {name} {ra_str} {dec_str} 2000.0 '
            f'0.000 0.000 {rot_angle:.1f} {rot_mode} '
            f'0 0 0.000 0 0 0.000 0.000'
        )

    lines.append('#')
    lines.append('# End of catalog')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info("Wrote Magellan catalog: %s (%d targets)", output_path, len(df))
    return output_path
