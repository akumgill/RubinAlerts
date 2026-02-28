"""Magellan telescope observing plan generation for SN Ia follow-up.

Computes merit scores for spectroscopic follow-up prioritization and
generates target catalogs in the Magellan TCS 16-field format.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
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

def compute_merit(delta_t, peak_mag,
                  tau=10.0, mag_optimal=20.5,
                  mag_bright=18.0, mag_faint=23.0):
    """Compute spectroscopic follow-up merit score.

    M = W_t(delta_t) * W_m(peak_mag)

    W_t = exp(-delta_t^2 / (2 * tau^2))        Gaussian time weight
    W_m = exp(-((m - m_opt) / sigma_m)^2)       Gaussian magnitude weight

    Parameters
    ----------
    delta_t : float or array-like
        Time since peak in days (MJD_now - peak_mjd). Sign preserved but
        the Gaussian uses delta_t^2, so it is symmetric.
    peak_mag : float or array-like
        Estimated peak apparent magnitude (AB).
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

    # NaN propagation is automatic from numpy, but be explicit
    merit = np.where(np.isfinite(delta_t) & np.isfinite(peak_mag),
                     merit, np.nan)
    return merit


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
                              max_airmass=2.0, min_hours_up=1.0):
    """Filter targets observable from Las Campanas on a given night.

    Parameters
    ----------
    targets_df : pd.DataFrame
        Must have 'ra' and 'dec' columns (degrees).
    obs_date : str
        UT date of observing night (e.g., '2026-03-15').
    site : EarthLocation
        Observatory location.
    max_airmass : float
        Maximum acceptable airmass (default 2.0 = 30 deg altitude).
    min_hours_up : float
        Minimum hours target must be above the airmass limit.

    Returns
    -------
    pd.DataFrame — copy of input filtered to observable targets, with
    added columns: transit_alt, min_airmass, hours_observable.
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

    # Compute altitude for each target at each time step
    transit_alts = np.full(len(targets_df), -90.0)
    hours_up = np.zeros(len(targets_df))
    dt_step = dark_hours / n_steps  # hours per step

    for t in times:
        frame = AltAz(obstime=t, location=site)
        alts = coords.transform_to(frame).alt.deg
        transit_alts = np.maximum(transit_alts, alts)
        hours_up += np.where(alts >= min_alt, dt_step, 0.0)

    out = targets_df.copy()
    out['transit_alt'] = transit_alts
    out['min_airmass'] = np.where(
        transit_alts > 0,
        1.0 / np.sin(np.radians(transit_alts)),
        np.inf,
    )
    out['hours_observable'] = hours_up

    # Filter
    mask = (hours_up >= min_hours_up)
    n_before = len(out)
    out = out[mask].copy()
    logger.info("Observability filter: %d/%d targets pass (airmass < %.1f, "
                ">= %.1f hours)", len(out), n_before, max_airmass, min_hours_up)

    return out


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
