"""Tests for load_targets_csv manual-target columns (chunk G: R15).

load_targets_csv now honors optional manual-workflow columns:
  - ``program``      -> Target.program (else default + a warning)
  - ``phase_weight`` -> Target.phase_weight directly
  - ``peak_mjd``     -> phase weight via the alert-pipeline Gaussian (needs
                        night_mjd)

Fully in-memory: tiny CSVs built in tmp_path; no network/DB.
"""

import logging

from orchestrator.normalize import (
    load_targets_csv, phase_weight_from_peak, PHASE_WEIGHT_TAU_DAYS,
)


def _write_csv(tmp_path, text):
    path = tmp_path / "targets.csv"
    path.write_text(text)
    return str(path)


def test_minimal_csv_still_loads(tmp_path):
    """A minimal name/ra/dec CSV (no optional columns) loads without crashing."""
    path = _write_csv(tmp_path, "name,ra,dec\nSN1,150.0,2.0\n")
    targets = load_targets_csv(path)
    assert len(targets) == 1
    t = targets[0]
    assert t.name == "SN1"
    assert t.program == "default"
    # Phase stays neutral (NaN) — not fabricated near-peak.
    assert t.phase_weight != t.phase_weight  # NaN


def test_explicit_program_honored(tmp_path):
    """A row WITH a program column is charged to that program."""
    path = _write_csv(
        tmp_path, "name,ra,dec,program\nSN1,150.0,2.0,MAGNETS-Stubbs\n")
    targets = load_targets_csv(path)
    assert targets[0].program == "MAGNETS-Stubbs"


def test_missing_program_warns_and_defaults(tmp_path, caplog):
    """A row WITHOUT a program falls back to the default AND emits a warning."""
    path = _write_csv(tmp_path, "name,ra,dec\nSN1,150.0,2.0\n")
    with caplog.at_level(logging.WARNING, logger="orchestrator.normalize"):
        targets = load_targets_csv(path, default_program="PROG-X")
    assert targets[0].program == "PROG-X"
    assert any("default program" in r.message for r in caplog.records)


def test_blank_program_cell_warns_and_defaults(tmp_path, caplog):
    """A present-but-blank program cell is treated as absent (warns + default)."""
    path = _write_csv(
        tmp_path, "name,ra,dec,program\nSN1,150.0,2.0,\n")
    with caplog.at_level(logging.WARNING, logger="orchestrator.normalize"):
        targets = load_targets_csv(path, default_program="PROG-X")
    assert targets[0].program == "PROG-X"
    assert any("default program" in r.message for r in caplog.records)


def test_explicit_phase_weight_honored(tmp_path):
    """An explicit phase_weight column is used directly."""
    path = _write_csv(
        tmp_path, "name,ra,dec,phase_weight\nSN1,150.0,2.0,0.42\n")
    targets = load_targets_csv(path)
    assert abs(targets[0].phase_weight - 0.42) < 1e-9


def test_peak_mjd_near_peak_beats_far(tmp_path):
    """peak_mjd near the night yields a higher phase weight than one far away."""
    night = 61000.0
    near = _write_csv(
        tmp_path, f"name,ra,dec,peak_mjd\nNEAR,150.0,2.0,{night}\n")
    near_targets = load_targets_csv(near, night_mjd=night)

    far = tmp_path / "far.csv"
    far.write_text(f"name,ra,dec,peak_mjd\nFAR,150.0,2.0,{night - 40.0}\n")
    far_targets = load_targets_csv(str(far), night_mjd=night)

    assert near_targets[0].phase_weight > far_targets[0].phase_weight
    # Exactly at peak -> w_time == 1.0.
    assert abs(near_targets[0].phase_weight - 1.0) < 1e-9


def test_explicit_phase_weight_wins_over_peak_mjd(tmp_path):
    """When both phase_weight and peak_mjd are present, phase_weight wins."""
    path = _write_csv(
        tmp_path,
        "name,ra,dec,phase_weight,peak_mjd\nSN1,150.0,2.0,0.5,61000.0\n")
    targets = load_targets_csv(path, night_mjd=61000.0)
    assert abs(targets[0].phase_weight - 0.5) < 1e-9


def test_peak_mjd_without_night_date_stays_neutral(tmp_path, caplog):
    """peak_mjd present but no night_mjd -> phase neutral (NaN) + a warning."""
    path = _write_csv(
        tmp_path, "name,ra,dec,peak_mjd\nSN1,150.0,2.0,61000.0\n")
    with caplog.at_level(logging.WARNING, logger="orchestrator.normalize"):
        targets = load_targets_csv(path)  # night_mjd defaults to NaN
    pw = targets[0].phase_weight
    assert pw != pw  # NaN
    assert any("night date" in r.message for r in caplog.records)


def test_no_phase_info_stays_neutral(tmp_path):
    """Neither phase_weight nor peak_mjd -> neutral NaN (no fabricated peak)."""
    path = _write_csv(tmp_path, "name,ra,dec\nSN1,150.0,2.0\n")
    targets = load_targets_csv(path)
    pw = targets[0].phase_weight
    assert pw != pw  # NaN


def test_phase_weight_from_peak_matches_gaussian():
    """Helper reproduces exp(-dt²/2τ²) with tau=10 d."""
    import math
    night, peak = 61000.0, 60990.0  # dt = 10 d = tau
    expected = math.exp(-(10.0 ** 2) / (2.0 * PHASE_WEIGHT_TAU_DAYS ** 2))
    assert abs(phase_weight_from_peak(peak, night) - expected) < 1e-12
    # Non-finite inputs -> NaN.
    nan = phase_weight_from_peak(float('nan'), night)
    assert nan != nan


def test_rubinalerts_program_column_routes_targets(tmp_path):
    """Classification-based routing: an optional 'program' column on
    candidates.csv assigns each alert target to its science program;
    blank/missing falls back to the default program."""
    import pandas as pd
    from orchestrator.normalize import load_from_rubinalerts

    df = pd.DataFrame([
        {'object_id': 'ZTFia', 'ra': 150.0, 'dec': -20.0, 'merit': 0.5,
         'program': 'MAGNETS-Ia'},
        {'object_id': 'ZTFslsn', 'ra': 151.0, 'dec': -21.0, 'merit': 0.4,
         'program': 'MAGNETS-Exotic'},
        {'object_id': 'DIAnew', 'ra': 152.0, 'dec': -22.0, 'merit': 0.3,
         'program': ''},
        {'object_id': 'DIAold', 'ra': 153.0, 'dec': -23.0, 'merit': 0.2,
         'program': None},
    ])
    path = tmp_path / 'candidates.csv'
    df.to_csv(path, index=False)
    targets = load_from_rubinalerts(str(path), default_program='MAGNETS-Ia')
    by_name = {t.name: t.program for t in targets}
    assert by_name['ZTFia'] == 'MAGNETS-Ia'
    assert by_name['ZTFslsn'] == 'MAGNETS-Exotic'
    assert by_name['DIAnew'] == 'MAGNETS-Ia'   # blank -> default
    assert by_name['DIAold'] == 'MAGNETS-Ia'   # missing -> default
