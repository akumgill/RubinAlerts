"""Rubin Observatory LSST Deep Drilling Field definitions."""

import numpy as np
from astropy.coordinates import SkyCoord, AltAz, get_sun
from astropy.time import Time
import astropy.units as u

# Search radius in degrees (LSST FOV is ~3.5 deg diameter)
DDF_SEARCH_RADIUS_DEG = 1.75

DDF_FIELDS = [
    {'name': 'COSMOS',   'ra': 150.11, 'dec':   2.23},
    {'name': 'XMM-LSS',  'ra':  35.57, 'dec':  -4.82},
    {'name': 'ECDFS',    'ra':  52.98, 'dec': -28.12},
    {'name': 'ELAIS-S1', 'ra':   9.45, 'dec': -44.02},
    {'name': 'EDFS_a',   'ra':  58.90, 'dec': -49.32},
    {'name': 'EDFS_b',   'ra':  63.60, 'dec': -47.60},
    {'name': 'M49',      'ra': 187.44, 'dec':   8.00},
]


def get_ddf_skycoords():
    """Return a list of (name, SkyCoord) tuples for all DDFs."""
    return [
        (f['name'], SkyCoord(ra=f['ra'] * u.deg, dec=f['dec'] * u.deg))
        for f in DDF_FIELDS
    ]


def is_in_ddf(ra, dec, radius_deg=DDF_SEARCH_RADIUS_DEG):
    """Check if a coordinate falls within any DDF footprint."""
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    for f in DDF_FIELDS:
        center = SkyCoord(ra=f['ra'] * u.deg, dec=f['dec'] * u.deg)
        if coord.separation(center).deg <= radius_deg:
            return f['name']
    return None


# ---------------------------------------------------------------------------
# Broker sky coverage
# ---------------------------------------------------------------------------

# Declination floor for ZTF-fed brokers. ZTF (and the ALeRCE-ZTF / ANTARES /
# Fink-ZTF streams it feeds) does not reach dec < -32 deg. See
# broker_clients/alerce_db_client.py crossmatch_positions(min_dec=-32.0).
ZTF_MIN_DEC = -32.0

# Maximum number of independent brokers that can ever report a transient,
# split by sky region:
#   dec > -32  : ZTF-fed brokers (ALeRCE-ZTF, ANTARES, Fink) + LSST-fed
#                brokers (ALeRCE-LSST, Fink-LSST). Up to 4 distinct brokers.
#   dec <= -32 : LSST-only — no ZTF coverage. Up to 2 distinct brokers
#                (ALeRCE-LSST, Fink-LSST).
# These caps bound the multi-broker agreement bonus so that southern DDFs,
# which can only ever be single/low-broker, are not penalised relative to
# equatorial fields purely because of where they sit on the sky.
MAX_BROKERS_ZTF_COVERAGE = 4
MAX_BROKERS_LSST_ONLY = 2


def field_visibility(date_str, location, airmass_limit=1.6,
                     twilight_deg=18.0, step_min=10):
    """Per-DDF observability report for a given UT night at ``location``.

    DDFs are autumn/winter southern-sky targets, so how well-placed each field
    is varies strongly with date. This samples the astronomical-night window
    and reports, per field, how low (airmass) and for how long it can be
    observed.

    Parameters
    ----------
    date_str : str
        UT date in YYYY-MM-DD format. The night sampled runs from
        ``date_str 20:00 UT`` for ~16 hours.
    location : astropy.coordinates.EarthLocation
        Observatory location (e.g. Las Campanas).
    airmass_limit : float, optional
        Airmass ceiling for the ``hours_below_limit`` accumulator (default 1.6).
    twilight_deg : float, optional
        Sun-altitude depression defining astronomical night (default 18 deg).
    step_min : float, optional
        Time-grid sampling step in minutes (default 10).

    Returns
    -------
    list of dict
        One dict per field, sorted by ``min_airmass`` ascending. Each dict has::

            name              field name (str)
            ra, dec           field centre in degrees (float)
            min_airmass       minimum airmass over the dark grid (float; inf if
                              the field never rises above the horizon)
            hours_below_limit hours with airmass <= airmass_limit (float)
            transit_ut        UT 'YYYY-MM-DD HH:MM' of minimum airmass (str or
                              None if never observable)
            dark_hours        length of the astronomical night in hours (float)
            well_placed       True if min_airmass < 1.5 and hours_below_limit > 0

        When there is no astronomical night (e.g. polar summer), every field is
        returned with dark_hours=0, min_airmass=inf and well_placed=False.

    Notes
    -----
    IERS auto-download is disabled here so the function never touches the
    network. Airmass at this precision (secz, ~minutes-scale grid) does not need
    fresh Earth-orientation data.
    """
    from astropy.utils import iers
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None

    # Sample a coarse time grid over the night in UT.
    start = Time(f"{date_str} 20:00:00", scale='utc')
    n_steps = int(round(16 * 60 / step_min)) + 1
    offsets = np.arange(n_steps) * step_min * u.min
    grid = start + offsets

    # Astronomical night: Sun altitude below -twilight_deg.
    frame = AltAz(obstime=grid, location=location)
    sun_alt = get_sun(grid).transform_to(frame).alt.deg
    dark_mask = sun_alt < -twilight_deg
    dark_grid = grid[dark_mask]

    dark_hours = float(len(dark_grid) * step_min / 60.0)

    results = []
    if len(dark_grid) == 0:
        for f in DDF_FIELDS:
            results.append({
                'name': f['name'], 'ra': f['ra'], 'dec': f['dec'],
                'min_airmass': float('inf'), 'hours_below_limit': 0.0,
                'transit_ut': None, 'dark_hours': 0.0, 'well_placed': False,
            })
        return sorted(results, key=lambda d: d['min_airmass'])

    dark_frame = AltAz(obstime=dark_grid, location=location)
    for f in DDF_FIELDS:
        coord = SkyCoord(ra=f['ra'] * u.deg, dec=f['dec'] * u.deg)
        altaz = coord.transform_to(dark_frame)
        # secz airmass; mask out below-horizon samples (secz negative/huge).
        above = altaz.alt.deg > 0
        airmass = np.full(len(dark_grid), np.inf)
        airmass[above] = altaz.secz.value[above]

        min_idx = int(np.argmin(airmass))
        min_airmass = float(airmass[min_idx])
        if np.isfinite(min_airmass):
            transit_ut = dark_grid[min_idx].iso[:16]
        else:
            transit_ut = None

        n_below = int(np.sum(airmass <= airmass_limit))
        hours_below_limit = float(n_below * step_min / 60.0)

        well_placed = bool(min_airmass < 1.5 and hours_below_limit > 0)

        results.append({
            'name': f['name'], 'ra': f['ra'], 'dec': f['dec'],
            'min_airmass': min_airmass, 'hours_below_limit': hours_below_limit,
            'transit_ut': transit_ut, 'dark_hours': dark_hours,
            'well_placed': well_placed,
        })

    return sorted(results, key=lambda d: d['min_airmass'])


def max_possible_brokers(dec):
    """Max number of brokers that can detect a transient at this declination.

    Parameters
    ----------
    dec : float or array-like
        Declination in degrees.

    Returns
    -------
    int or numpy array
        MAX_BROKERS_ZTF_COVERAGE where dec > ZTF_MIN_DEC (ZTF-fed brokers
        reachable), otherwise MAX_BROKERS_LSST_ONLY.
    """
    import numpy as np
    dec_arr = np.asarray(dec, dtype=float)
    result = np.where(dec_arr > ZTF_MIN_DEC,
                      MAX_BROKERS_ZTF_COVERAGE,
                      MAX_BROKERS_LSST_ONLY)
    if np.ndim(dec) == 0:
        return int(result)
    return result.astype(int)
