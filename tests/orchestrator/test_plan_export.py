"""Exports of the observing plan (catalog / CSV / print sheet).

Focus: the sub-exposure breakdown ('N x Y s') is carried into the observer's
sheets — a submitter's requested cadence must not be lost — and the three
formats render without error.
"""
from api import plan_export


def _dash(target_extra=None):
    tgt = {"name": "SN-A", "obs_start": "01:00", "obs_end": "05:00",
           "obs_best": "03:00", "min_airmass": 1.1, "window_note": "up most of the night",
           "notes": "bumpy IIn"}
    tgt.update(target_extra or {})
    return {
        "plan": {"date": "2026-08-13", "instrument": "LDSS3", "moon": "grey",
                 "n_scheduled": 1, "twilight_start": "00:30", "twilight_end": "09:00",
                 "timeline": [{"utc": "01:00-01:30", "target": "SN-A",
                               "program": "CfA-Villar", "tier": "P1",
                               "ra": 10.0, "dec": -5.0, "mag": 19.0, "exp_min": 30}]},
        "targets": [tgt],
    }


def test_submitter_subexposure_spec_survives_to_sheets():
    dash = _dash({"n_exposures": 3, "exposure_seconds": 600})
    csv = plan_export.observing_csv(dash)
    txt = plan_export.observing_text(dash)
    assert "exp_spec" in csv.splitlines()[0]      # header present
    assert "3x600s" in csv
    assert "3x600s" in txt


def test_subexposure_derived_from_total_when_unspecified():
    # No submitter spec -> a cosmic-ray split is derived from the 30-min total.
    csv = plan_export.observing_csv(_dash())
    assert "x300s" in csv          # split_exposure default 300 s subs


def test_all_three_formats_render():
    dash = _dash({"n_exposures": 2, "exposure_seconds": 900})
    assert "SN-A" in plan_export.catalog_text(dash)
    assert "SN-A" in plan_export.observing_csv(dash)
    assert "SN-A" in plan_export.observing_text(dash)
