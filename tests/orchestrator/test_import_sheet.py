"""The collaboration spreadsheet is the real submission interface.

These pin the sheet's own conventions — free-text magnitudes with bands and
surface brightnesses, 'NxMs' exposures, a 1-5 priority scale, and a person in
Requested By — so a change in the workbook shows up as a failing test rather
than as a silently mis-sized exposure.
"""
import importlib.util
import os
import sys

import pandas as pd
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mod():
    path = os.path.join(_REPO, "scripts", "import_sheet.py")
    spec = importlib.util.spec_from_file_location("import_sheet", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_parse_mag_keeps_band_and_spots_surface_brightness():
    m = _mod()
    # the four shapes the real sheet actually contains
    assert m.parse_mag("22 in r") == (22.0, "r", "point", True)
    assert m.parse_mag("20.5 in g") == (20.5, "g", "point", True)
    sb = m.parse_mag('22.1 mag arcsec⁻² in r at Re = 3.6"')
    assert sb[0] == pytest.approx(22.1) and sb[1] == "r"
    assert sb[2] == "surface_brightness"
    # no band stated -> defaults to r but SAYS it was assumed, so the caller
    # can warn instead of quietly feeding an unknown band to an r-indexed ETC
    val, band, kind, said = m.parse_mag("20 mag/arcs^2")
    assert (val, band, kind, said) == (20.0, "r", "surface_brightness", False)
    assert m.parse_mag("")[0] != m.parse_mag("")[0]        # nan


def test_parse_exposure_and_priority():
    m = _mod()
    assert m.parse_exposure("3x1200s") == (3, 1200.0)
    assert m.parse_exposure("5 X 900") == (5, 900.0)
    assert m.parse_exposure("whenever")[0] is None
    assert m.parse_priority(1) == "P1" and m.parse_priority("4") == "P4"
    # P0 is our tonight-guarantee and must never be inferred from a sheet
    assert m.parse_priority(0) == "P3"
    assert m.parse_priority("junk") == "P3"


def test_rows_to_items_maps_the_real_column_names():
    m = _mod()
    df = pd.DataFrame([{
        "Target": "2023mfm", "RA (J2000 deg)": 324.366355,
        "Dec (J2000 deg)": -4.345197,
        "Priority \n(1-5; if 4 or 5, added to backup plan)": 1,
        "Apparent \nMag & Band": '22.1 mag arcsec⁻² in r at Re = 3.6"',
        "Exposure Time": "3x1200s", "Requested By": "Wenkai (UA)",
        "Description \n(e.g, \"interacting SNIc\")": "Off-nuclear TDE host",
        "link": "https://example.org/x",
    }])
    items, warnings = m.rows_to_items(df, instrument="LLAMAS")
    assert len(items) == 1
    it = items[0]
    assert it["name"] == "2023mfm" and it["priority"] == "P1"
    assert it["mag_kind"] == "surface_brightness"
    assert (it["n_exposures"], it["exposure_seconds"]) == (3, 1200.0)
    # the PERSON is carried; the program comes from the API key, not the sheet
    assert it["requested_by"] == "Wenkai (UA)" and "program" not in it
    assert it["link"] == "https://example.org/x"
    assert any("SURFACE BRIGHTNESS" in w for w in warnings)


def test_missing_required_column_is_an_error_not_a_silent_skip():
    m = _mod()
    with pytest.raises(ValueError, match="exposure"):
        m.rows_to_items(pd.DataFrame([{"Target": "x", "Priority": 1}]))
