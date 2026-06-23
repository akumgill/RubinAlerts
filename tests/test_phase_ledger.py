"""Tests for the phase-split integration ledger and multi-group alerts (W12).

All offline: synthetic Targets, temp-file JSON/YAML only. Nothing here contacts
a broker, database, or external API.
"""

import json
import math
from pathlib import Path

import pytest

from orchestrator.models import Target
from orchestrator.config import LLAMAS_CONFIG
from orchestrator.accounting import TimeAccountant
from orchestrator.target_ledger import (
    TargetLedger, TargetLedgerEntry, phase_bucket,
)
from orchestrator.run_nightly import detect_multi_group_alerts


DATE = '2026-10-15'


def _target(name, ra=150.0, dec=2.0, priority=1, program='default',
            delta_t=float('nan')):
    return Target(name=name, ra_deg=ra, dec_deg=dec, priority=priority,
                  program=program, delta_t=delta_t)


# ---------------------------------------------------------------------------
# phase_bucket boundaries
# ---------------------------------------------------------------------------

def test_phase_bucket_boundaries():
    w = LLAMAS_CONFIG.phase_bucket_window_days  # 5.0
    assert phase_bucket(-10.0, w) == 'rising'
    assert phase_bucket(-w - 0.1, w) == 'rising'
    assert phase_bucket(-w, w) == 'peak'      # |dt| <= W -> peak
    assert phase_bucket(0.0, w) == 'peak'
    assert phase_bucket(w, w) == 'peak'
    assert phase_bucket(w + 0.1, w) == 'declining'
    assert phase_bucket(10.0, w) == 'declining'


def test_phase_bucket_nan_and_none():
    assert phase_bucket(float('nan'), 5.0) == 'all'
    assert phase_bucket(None, 5.0) == 'all'


# ---------------------------------------------------------------------------
# Per-phase satisfaction does not cross buckets
# ---------------------------------------------------------------------------

def test_rising_time_does_not_satisfy_peak(tmp_path):
    """Charging rising-phase time does NOT satisfy the 'peak' bucket."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t = _target('T')
    req = 60.0  # minutes

    # Fully integrate the RISING bucket.
    ledger.charge(t, science_seconds=req * 60.0, date=DATE,
                  required_seconds=req * 60.0, phase='rising')

    assert ledger.is_satisfied(t, req, phase='rising') is True
    assert ledger.is_satisfied(t, req, phase='peak') is False
    # Peak bucket has zero accumulated.
    assert ledger.cumulative_seconds(t, phase='peak') == 0.0
    assert ledger.remaining_minutes(t, req, phase='peak') == pytest.approx(req)
    # Total across buckets still reflects the rising time.
    assert ledger.cumulative_seconds(t) == pytest.approx(req * 60.0)


# ---------------------------------------------------------------------------
# Backward compatibility: phase omitted == summed total
# ---------------------------------------------------------------------------

def test_phase_omitted_uses_summed_total(tmp_path):
    """With phase omitted, queries use the SUM across buckets."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t = _target('T')
    ledger.charge(t, science_seconds=20 * 60.0, date=DATE, phase='peak')
    ledger.charge(t, science_seconds=10 * 60.0, date=DATE, phase='rising')
    # Summed total.
    assert ledger.cumulative_seconds(t) == pytest.approx(30 * 60.0)
    assert ledger.remaining_minutes(t, 60.0) == pytest.approx(30.0)
    # Per-bucket.
    assert ledger.cumulative_seconds(t, phase='peak') == pytest.approx(20 * 60.0)
    assert ledger.cumulative_seconds(t, phase='rising') == pytest.approx(10 * 60.0)


def test_cumulative_science_seconds_accessor():
    """The scalar accessor sums the per-phase buckets."""
    ent = TargetLedgerEntry(
        cumulative_seconds_by_phase={'peak': 600.0, 'rising': 300.0})
    assert ent.cumulative_science_seconds == pytest.approx(900.0)


# ---------------------------------------------------------------------------
# Old-format migration + round-trip
# ---------------------------------------------------------------------------

def test_old_format_scalar_migrates_to_all_bucket(tmp_path):
    """An old-format ledger JSON (scalar cumulative_science_seconds) migrates
    into the 'all' bucket on load and round-trips."""
    state_path = tmp_path / 'old_ledger.json'
    old_state = {
        'entries': {
            '150.00000_+2.00000': {
                'coord_key': '150.00000_+2.00000',
                'canonical_name': 'SN_OLD',
                'aliases': ['SN_OLD'],
                'ra_deg': 150.0,
                'dec_deg': 2.0,
                'cumulative_science_seconds': 1800.0,
                'required_seconds_history': [],
                'nights_observed': ['2026-01-01'],
                'charge_log': [],
            }
        }
    }
    state_path.write_text(json.dumps(old_state))

    ledger = TargetLedger.load(str(state_path))
    t = _target('SN_OLD', ra=150.0, dec=2.0)
    # Migrated into 'all'.
    assert ledger.cumulative_seconds(t, phase='all') == pytest.approx(1800.0)
    assert ledger.cumulative_seconds(t) == pytest.approx(1800.0)

    # Round-trip: persist then reload.
    ledger.charge(t, science_seconds=600.0, date=DATE, phase='peak')
    reloaded = TargetLedger.load(str(state_path))
    assert reloaded.cumulative_seconds(t, phase='all') == pytest.approx(1800.0)
    assert reloaded.cumulative_seconds(t, phase='peak') == pytest.approx(600.0)
    assert reloaded.cumulative_seconds(t) == pytest.approx(2400.0)


def test_charge_records_program(tmp_path):
    """charge() records the charging program on the entry."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t = _target('T', program='MAGNETS-Stubbs')
    ledger.charge(t, science_seconds=600.0, date=DATE, phase='peak',
                  program='MAGNETS-Stubbs')
    ent = ledger._find_entry(t)
    assert ent.programs == ['MAGNETS-Stubbs']
    # A second charge by the same program is not duplicated.
    ledger.charge(t, science_seconds=600.0, date=DATE, phase='peak',
                  program='MAGNETS-Stubbs')
    assert ent.programs == ['MAGNETS-Stubbs']


def test_per_phase_reconcile(tmp_path):
    """A per-phase reconcile only true-ups that bucket."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t = _target('T')
    ledger.charge(t, science_seconds=30 * 60.0, date=DATE, phase='peak')
    delta = ledger.reconcile(t, actual_seconds=22 * 60.0, date=DATE,
                             phase='peak')
    assert delta == pytest.approx(-8 * 60.0)
    assert ledger.cumulative_seconds(t, phase='peak') == pytest.approx(22 * 60.0)


# ---------------------------------------------------------------------------
# multi_program_entries
# ---------------------------------------------------------------------------

def test_multi_program_entries(tmp_path):
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    shared = _target('SHARED', ra=150.0, dec=2.0)
    ledger.charge(shared, science_seconds=600.0, date=DATE, program='A')
    ledger.charge(shared, science_seconds=600.0, date=DATE, program='B')
    solo = _target('SOLO', ra=10.0, dec=-20.0)
    ledger.charge(solo, science_seconds=600.0, date=DATE, program='A')

    multi = ledger.multi_program_entries()
    assert len(multi) == 1
    assert multi[0].canonical_name == 'SHARED'


# ---------------------------------------------------------------------------
# Multi-group alert detection
# ---------------------------------------------------------------------------

def test_multi_group_two_targets_same_coords_different_programs(
        tmp_path, sample_allocations_path):
    """Two targets at the same position with DIFFERENT programs -> one alert,
    with same_phase reflecting their preferences."""
    accountant = TimeAccountant.from_yaml(sample_allocations_path)
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t_peak = _target('OBJ_S', ra=150.0, dec=2.0, program='MAGNETS-Stubbs',
                     delta_t=0.0)
    t_rise = _target('OBJ_V', ra=150.00005, dec=2.00005,
                     program='MAGNETS-Villar', delta_t=-7.0)

    alerts = detect_multi_group_alerts([t_peak, t_rise], ledger,
                                       accountant=accountant)
    assert len(alerts) == 1
    a = alerts[0]
    assert set(a['programs']) == {'MAGNETS-Stubbs', 'MAGNETS-Villar'}
    # Stubbs=peak, Villar=rising -> different phase preferences.
    assert a['same_phase'] is False
    assert a['phase_preferences']['MAGNETS-Stubbs'] == 'peak'
    assert a['phase_preferences']['MAGNETS-Villar'] == 'rising'


def test_no_alert_same_program(tmp_path, sample_allocations_path):
    """Two targets at the same position with the SAME program -> no alert."""
    accountant = TimeAccountant.from_yaml(sample_allocations_path)
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t1 = _target('OBJ_A', ra=150.0, dec=2.0, program='MAGNETS-Stubbs')
    t2 = _target('OBJ_B', ra=150.00005, dec=2.00005, program='MAGNETS-Stubbs')
    alerts = detect_multi_group_alerts([t1, t2], ledger, accountant=accountant)
    assert alerts == []


def test_alert_from_prior_ledger_program(tmp_path, sample_allocations_path):
    """A ledger entry already charged by program A, then a tonight target for
    program B at the same coords -> alert."""
    accountant = TimeAccountant.from_yaml(sample_allocations_path)
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))

    # Prior night: program A (Stubbs) charged this position.
    prior = _target('OBJ', ra=150.0, dec=2.0, program='MAGNETS-Stubbs')
    ledger.charge(prior, science_seconds=600.0, date='2026-10-01',
                  phase='peak', program='MAGNETS-Stubbs')

    # Tonight: program B (Villar) wants the same object.
    tonight = _target('OBJ', ra=150.00005, dec=2.00005,
                      program='MAGNETS-Villar', delta_t=-7.0)
    alerts = detect_multi_group_alerts([tonight], ledger,
                                       accountant=accountant)
    assert len(alerts) == 1
    assert set(alerts[0]['programs']) == {'MAGNETS-Stubbs', 'MAGNETS-Villar'}
    assert alerts[0]['same_phase'] is False


# ---------------------------------------------------------------------------
# run-nightly-style charge flow: correct bucket + program from delta_t
# ---------------------------------------------------------------------------

def test_charge_flow_buckets_from_delta_t(tmp_path):
    """A run-nightly-style flow charges the bucket implied by delta_t and
    records the program."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    w = LLAMAS_CONFIG.phase_bucket_window_days

    rising = _target('RISE', ra=150.0, dec=2.0, program='MAGNETS-Villar',
                     delta_t=-9.0)
    bucket = phase_bucket(rising.delta_t, w)
    assert bucket == 'rising'
    ledger.charge(rising, science_seconds=30 * 60.0, date=DATE,
                  phase=bucket, program=rising.program)

    ent = ledger._find_entry(rising)
    assert set(ent.cumulative_seconds_by_phase) == {'rising'}
    assert ent.cumulative_seconds_by_phase['rising'] == pytest.approx(30 * 60.0)
    assert ent.programs == ['MAGNETS-Villar']
    # charge_log records the phase + program.
    log = ent.charge_log[-1]
    assert log['phase'] == 'rising'
    assert log['program'] == 'MAGNETS-Villar'
