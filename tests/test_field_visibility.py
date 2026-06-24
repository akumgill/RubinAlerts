"""Tests for the per-night DDF visibility report (core.ddf_fields.field_visibility).

DDFs are autumn/winter southern-sky targets, so which fields are well-placed
swings strongly with date. These tests are deterministic and offline — the
function disables IERS auto-download, so airmass is computed from built-in
Earth-orientation data with no network access.
"""

import astropy.units as u
from astropy.coordinates import EarthLocation

from core.ddf_fields import field_visibility

# Las Campanas Observatory (LCO) — matches orchestrator.config.LLAMASConfig.
LCO = EarthLocation(lat=-29.0146 * u.deg, lon=-70.6926 * u.deg, height=2380 * u.m)


def _by_name(report):
    return {f['name']: f for f in report}


def test_august_southern_fields_well_placed():
    """2026-08-13: deep-southern fields ride high, COSMOS is poorly placed."""
    report = field_visibility('2026-08-13', LCO, airmass_limit=1.6)
    fields = _by_name(report)

    for name in ('ELAIS-S1', 'ECDFS', 'EDFS_a'):
        f = fields[name]
        assert f['well_placed'], f"{name} should be well-placed in August"
        assert f['min_airmass'] < 1.2, f"{name} min_airmass={f['min_airmass']}"
        assert f['hours_below_limit'] > 0

    cosmos = fields['COSMOS']
    assert not cosmos['well_placed'], "COSMOS should be poorly placed in August"
    assert cosmos['min_airmass'] > fields['ECDFS']['min_airmass']


def test_january_cosmos_returns():
    """2027-01-12: the equatorial COSMOS field is well-placed again."""
    report = field_visibility('2027-01-12', LCO, airmass_limit=1.6)
    cosmos = _by_name(report)['COSMOS']
    assert cosmos['well_placed']
    assert cosmos['min_airmass'] < 1.3


def test_report_sanity():
    """Common invariants: airmass >= 1, sorted ascending, real night."""
    for date in ('2026-08-13', '2027-01-12'):
        report = field_visibility(date, LCO, airmass_limit=1.6)
        airmasses = [f['min_airmass'] for f in report]

        assert all(a >= 1.0 for a in airmasses), f"{date}: airmass < 1"
        assert airmasses == sorted(airmasses), f"{date}: not sorted ascending"
        assert all(f['dark_hours'] > 0 for f in report), f"{date}: no dark night"
