"""Tests for the per-target integration ledger (W11 core).

All offline: synthetic Targets and temp-file JSON only. Nothing here contacts a
broker, database, or external API.
"""

import json
import math
from pathlib import Path

import pytest

from orchestrator.models import Target
from orchestrator.target_ledger import TargetLedger, TargetLedgerEntry
from orchestrator.prioritizer import compute_composite_score


DATE = '2026-10-15'


def _target(name, ra=150.0, dec=2.0, priority=1, notes=''):
    return Target(name=name, ra_deg=ra, dec_deg=dec, priority=priority,
                  notes=notes)


# ---------------------------------------------------------------------------
# Coordinate matching across name changes
# ---------------------------------------------------------------------------

def test_coordinate_match_across_name_change(tmp_path):
    """Same sky position, different name (internal id -> TNS) -> one entry,
    canonical name upgraded to the TNS designation, old name kept as alias."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t1 = _target('ALERT_0001', ra=150.0, dec=2.0)
    ledger.charge(t1, science_seconds=600.0, date=DATE,
                  required_seconds=3600.0)

    # Next night: same position (tiny offset), renamed to a TNS designation.
    t2 = _target('SN2026abc', ra=150.00005, dec=2.00005)
    ent = ledger.get_or_create(t2)

    assert len(ledger.entries) == 1
    assert ent.canonical_name == 'SN2026abc'
    assert 'ALERT_0001' in ent.aliases
    assert 'SN2026abc' in ent.aliases


def test_no_false_match_at_10_arcsec(tmp_path):
    """Targets 10\" apart are distinct objects -> two entries."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t1 = _target('A', ra=150.0, dec=2.0)
    ledger.charge(t1, science_seconds=100.0, date=DATE)

    # 10 arcsec = 10/3600 deg in dec.
    t2 = _target('B', ra=150.0, dec=2.0 + 10.0 / 3600.0)
    ledger.get_or_create(t2)
    assert len(ledger.entries) == 2


def test_match_boundary_1p9_matches_2p1_does_not(tmp_path):
    """Default 2.0\" radius: 1.9\" away matches, 2.1\" away does not."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    base = _target('BASE', ra=150.0, dec=2.0)
    ledger.charge(base, science_seconds=100.0, date=DATE)

    near = _target('NEAR', ra=150.0, dec=2.0 + 1.9 / 3600.0)
    assert ledger._find_entry(near) is not None

    far = _target('FAR', ra=150.0, dec=2.0 + 2.1 / 3600.0)
    assert ledger._find_entry(far) is None


# ---------------------------------------------------------------------------
# Accumulation / satisfaction
# ---------------------------------------------------------------------------

def test_accumulation_crosses_satisfied_fraction(tmp_path):
    """Charging past 0.95 of required marks the target satisfied."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t = _target('T')
    required_min = 120.0  # 7200 s

    # 60 min -> 0.5; remaining 60 min >> min_block -> not satisfied.
    ledger.charge(t, science_seconds=60 * 60.0, date='2026-10-15',
                  required_seconds=required_min * 60.0)
    assert not ledger.is_satisfied(t, required_min)

    # +55 min -> 115/120 = 0.958 >= 0.95 -> satisfied.
    ledger.charge(t, science_seconds=55 * 60.0, date='2026-10-16',
                  required_seconds=required_min * 60.0)
    assert ledger.is_satisfied(t, required_min)


def test_remaining_minutes(tmp_path):
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t = _target('T')
    ledger.charge(t, science_seconds=20 * 60.0, date=DATE)
    assert ledger.remaining_minutes(t, 60.0) == pytest.approx(40.0)
    # Never negative.
    assert ledger.remaining_minutes(t, 10.0) == 0.0


def test_min_block_floor_satisfies(tmp_path):
    """85/95 min is below 0.95 but the 10-min remainder < min_block (15) -> the
    leftover isn't worth a block, so it counts as satisfied."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t = _target('T')
    required_min = 95.0
    ledger.charge(t, science_seconds=85 * 60.0, date=DATE,
                  required_seconds=required_min * 60.0)
    frac = ledger.completeness_fraction(t, required_min)
    assert frac < ledger.satisfied_fraction  # 0.895 < 0.95
    assert ledger.remaining_minutes(t, required_min) < ledger.min_block_minutes
    assert ledger.is_satisfied(t, required_min)


# ---------------------------------------------------------------------------
# Completeness factor monotonicity + floor
# ---------------------------------------------------------------------------

def test_completeness_factor_monotonicity_and_floor(tmp_path):
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t = _target('T')
    required_min = 100.0  # 6000 s

    # 0.0 done -> 1.0
    assert ledger.completeness_factor(t, required_min) == 1.0

    # 0.5 done -> 0.5
    ledger.charge(t, science_seconds=50 * 60.0, date='d1',
                  required_seconds=required_min * 60.0)
    assert ledger.completeness_factor(t, required_min) == pytest.approx(0.5)

    # 0.9 done -> max(min_factor=0.15, 0.1) = 0.15 (floor)
    ledger.charge(t, science_seconds=40 * 60.0, date='d2',
                  required_seconds=required_min * 60.0)
    assert ledger.completeness_factor(t, required_min) == pytest.approx(0.15)

    # 1.0 done -> 0.0
    ledger.charge(t, science_seconds=10 * 60.0, date='d3',
                  required_seconds=required_min * 60.0)
    assert ledger.completeness_factor(t, required_min) == 0.0


# ---------------------------------------------------------------------------
# Required recomputed on fade
# ---------------------------------------------------------------------------

def test_required_recomputed_on_fade(tmp_path):
    """Near-satisfied at one required time, but a fainter later epoch needs a
    LARGER required time -> no longer satisfied. The ledger stores cumulative
    seconds; satisfaction is judged against whatever required_minutes the caller
    passes (recomputed per night from the current magnitude)."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t = _target('T')

    # Night 1: mag 20 -> required 45 min; observe 44 -> nearly done.
    req1 = 45.0
    ledger.charge(t, science_seconds=44 * 60.0, date='n1',
                  mag=20.0, required_seconds=req1 * 60.0)
    assert ledger.is_satisfied(t, req1)  # 44/45 >= 0.95

    # Night 2: target faded to mag 21 -> required grows to ~112 min.
    req2 = 112.0
    assert not ledger.is_satisfied(t, req2)  # 44/112 = 0.39, remaining huge
    assert ledger.remaining_minutes(t, req2) == pytest.approx(68.0)


# ---------------------------------------------------------------------------
# Reconcile true-up
# ---------------------------------------------------------------------------

def test_reconcile_true_up(tmp_path):
    """Scheduled 30 min; actually integrated 22 -> reconcile adjusts cumulative
    down by the 8-min delta."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    t = _target('T')
    ledger.charge(t, science_seconds=30 * 60.0, date=DATE,
                  required_seconds=60 * 60.0)
    assert ledger.cumulative_seconds(t) == pytest.approx(30 * 60.0)

    delta = ledger.reconcile(t, actual_seconds=22 * 60.0, date=DATE)
    assert delta == pytest.approx(-8 * 60.0)
    assert ledger.cumulative_seconds(t) == pytest.approx(22 * 60.0)

    # Re-reconcile to the same actual -> no further change.
    assert ledger.reconcile(t, actual_seconds=22 * 60.0, date=DATE) == 0.0


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

def test_persistence_round_trip(tmp_path):
    state_path = str(tmp_path / 'ledger.json')
    ledger = TargetLedger(state_path=state_path)
    t = _target('SN2026xyz', ra=150.0, dec=2.0)
    ledger.charge(t, science_seconds=1200.0, date=DATE,
                  mag=20.5, redshift=0.1, required_seconds=3600.0)

    assert Path(state_path).exists()

    reloaded = TargetLedger.load(state_path)
    assert len(reloaded.entries) == 1
    assert reloaded.cumulative_seconds(t) == pytest.approx(1200.0)
    ent = reloaded._find_entry(t)
    assert ent.canonical_name == 'SN2026xyz'
    # Required-exposure history survived the round-trip.
    hist = ent.required_seconds_history[-1]
    assert hist['required_s'] == pytest.approx(3600.0)
    assert hist['mag'] == pytest.approx(20.5)
    assert hist['redshift'] == pytest.approx(0.1)


def test_load_missing_file_is_empty(tmp_path):
    """A first-ever night: no ledger file -> empty ledger, no error."""
    ledger = TargetLedger.load(str(tmp_path / 'does_not_exist.json'))
    assert ledger.entries == {}


# ---------------------------------------------------------------------------
# Prioritizer integration: completeness folded into the score
# ---------------------------------------------------------------------------

def test_satisfied_p1_ranks_below_fresh_p4(tmp_path):
    """A satisfied (completeness 0) high-priority P1 scores below a fresh P4:
    completeness 0 zeroes the science term."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    required_min = 60.0

    p1 = _target('P1', ra=150.0, dec=2.0, priority=1)
    # Fully integrate the P1 -> completeness factor 0.
    ledger.charge(p1, science_seconds=required_min * 60.0, date=DATE,
                  required_seconds=required_min * 60.0)
    p1_complete = ledger.completeness_factor(p1, required_min)
    assert p1_complete == 0.0

    p4 = _target('P4', ra=10.0, dec=-20.0, priority=4)
    p4_complete = ledger.completeness_factor(p4, required_min)  # fresh -> 1.0

    s_p1, bd_p1 = compute_composite_score(p1, completeness=p1_complete)
    s_p4, bd_p4 = compute_composite_score(p4, completeness=p4_complete)

    assert bd_p1['science_term'] == 0.0
    assert s_p4 > s_p1


def test_partially_done_p1_beats_filler_via_floor(tmp_path):
    """A 0.9-done P1 still beats nothing-special filler: completeness floors at
    min_factor (0.15) so the P1 science core (1.0*0.15=0.15*100=15) tops a P4
    filler with a demotion keyword."""
    ledger = TargetLedger(state_path=str(tmp_path / 'ledger.json'))
    required_min = 100.0

    p1 = _target('P1', ra=150.0, dec=2.0, priority=1)
    ledger.charge(p1, science_seconds=90 * 60.0, date=DATE,
                  required_seconds=required_min * 60.0)
    p1_complete = ledger.completeness_factor(p1, required_min)
    assert p1_complete == pytest.approx(0.15)

    filler = _target('FILLER', ra=10.0, dec=-20.0, priority=4,
                     notes='backup filler')
    s_p1, _ = compute_composite_score(p1, completeness=p1_complete)
    s_filler, _ = compute_composite_score(filler, completeness=1.0)
    assert s_p1 > s_filler


# ---------------------------------------------------------------------------
# Backward compatibility / invariant
# ---------------------------------------------------------------------------

def test_completeness_one_matches_prior_behavior():
    """compute_composite_score with completeness=1.0 (default) is unchanged."""
    t = _target('T', priority=2, notes='urgent')
    s_default, bd_default = compute_composite_score(t)
    s_explicit, bd_explicit = compute_composite_score(t, completeness=1.0)
    assert s_default == s_explicit
    assert bd_default['completeness'] == 1.0


def test_breakdown_invariant_holds_with_completeness():
    """total == science_term + observability_term + keyword_term, even with a
    non-trivial completeness factor folded into the science core."""
    t = _target('T', priority=1, notes='high priority')
    for c in (0.0, 0.15, 0.5, 1.0):
        _, bd = compute_composite_score(t, completeness=c)
        assert bd['total'] == pytest.approx(
            bd['science_term'] + bd['observability_term'] + bd['keyword_term'])
        # science_term reflects the completeness factor.
        assert bd['completeness'] == pytest.approx(c)
