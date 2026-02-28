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
