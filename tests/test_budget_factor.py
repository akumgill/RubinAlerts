"""Tests for phase-aware budget factor (chunk D: R11).

Uses the sample allocations fixture; state is temp-file JSON only.
MAGNETS-Stubbs allocation: dark 5h, grey 20h, bright 5h.
"""

from orchestrator.accounting import TimeAccountant
from orchestrator.cli import _default_state_path


def test_grey_exhausted_dark_full(sample_allocations_path, tmp_output_dir):
    """Grey exhausted but dark full → ~0.1 on a grey night, ~1.0 on a dark night."""
    state_path = _default_state_path(tmp_output_dir)
    acct = TimeAccountant.from_yaml(sample_allocations_path, state_path=state_path)

    # Burn all grey (20h) but leave dark untouched.
    acct.charge('MAGNETS-Stubbs', 20.0, 'grey', date='2026-10-15')

    assert abs(acct.get_budget_factor('MAGNETS-Stubbs', 'grey') - 0.1) < 1e-9
    assert abs(acct.get_budget_factor('MAGNETS-Stubbs', 'dark') - 1.0) < 1e-9


def test_fractional_thresholds(sample_allocations_path, tmp_output_dir):
    """>50% of phase allocation → 1.0; some left → 0.5."""
    state_path = _default_state_path(tmp_output_dir)
    acct = TimeAccountant.from_yaml(sample_allocations_path, state_path=state_path)

    # 60% of grey (20h) remaining: use 8h → 12h left = 0.6 fraction → 1.0.
    acct.charge('MAGNETS-Stubbs', 8.0, 'grey', date='2026-10-15')
    assert abs(acct.get_budget_factor('MAGNETS-Stubbs', 'grey') - 1.0) < 1e-9

    # Push to 10% remaining: use 10 more (18h total) → 2h left = 0.1 → 0.5.
    acct.charge('MAGNETS-Stubbs', 10.0, 'grey', date='2026-10-16')
    assert abs(acct.get_budget_factor('MAGNETS-Stubbs', 'grey') - 0.5) < 1e-9


def test_exactly_half_is_not_above_half(sample_allocations_path, tmp_output_dir):
    """Threshold is strictly > 0.5: exactly 50% remaining → 0.5, not 1.0."""
    state_path = _default_state_path(tmp_output_dir)
    acct = TimeAccountant.from_yaml(sample_allocations_path, state_path=state_path)

    acct.charge('MAGNETS-Stubbs', 10.0, 'grey', date='2026-10-15')  # 10/20 left
    assert abs(acct.get_budget_factor('MAGNETS-Stubbs', 'grey') - 0.5) < 1e-9


def test_zero_phase_allocation_is_exhausted(sample_allocations_path, tmp_output_dir):
    """A phase with 0 allocation → treated as exhausted (0.1), no div-by-zero."""
    state_path = _default_state_path(tmp_output_dir)
    acct = TimeAccountant.from_yaml(sample_allocations_path, state_path=state_path)

    # Manually zero out bright to simulate no bright allocation.
    acct.allocations['MAGNETS-Stubbs'].allocated_hours['bright'] = 0.0
    assert abs(acct.get_budget_factor('MAGNETS-Stubbs', 'bright') - 0.1) < 1e-9


def test_unknown_program_neutral(sample_allocations_path, tmp_output_dir):
    """Unknown program scores neutral 1.0 (preserved behavior)."""
    state_path = _default_state_path(tmp_output_dir)
    acct = TimeAccountant.from_yaml(sample_allocations_path, state_path=state_path)
    assert abs(acct.get_budget_factor('NOT-A-PROGRAM', 'grey') - 1.0) < 1e-9


def test_no_moon_phase_falls_back_to_total(sample_allocations_path, tmp_output_dir):
    """Without a moon phase, uses total remaining vs total allocation."""
    state_path = _default_state_path(tmp_output_dir)
    acct = TimeAccountant.from_yaml(sample_allocations_path, state_path=state_path)
    # Stubbs total alloc = 30h, all remaining → fraction 1.0 → 1.0.
    assert abs(acct.get_budget_factor('MAGNETS-Stubbs') - 1.0) < 1e-9
