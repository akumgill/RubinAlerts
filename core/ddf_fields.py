"""Rubin Observatory LSST Deep Drilling Field definitions."""

from astropy.coordinates import SkyCoord
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
