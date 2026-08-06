"""Tests for time-accounting correctness (chunk D: R3, R4).

All state is temp-file JSON; nothing here contacts a broker/DB/API.
"""

import json
from pathlib import Path

from astropy.time import Time
import astropy.units as u

from orchestrator.accounting import TimeAccountant
from orchestrator.cli import _default_state_path
from orchestrator.models import ScheduledEntry, Target


# ---------------------------------------------------------------------------
# W5 / R3 — unified state path, no reconcile double-charge
# ---------------------------------------------------------------------------

def test_default_state_path_matches_run_nightly_convention(tmp_output_dir):
    """reconcile and run-nightly must resolve to the same file."""
    resolved = _default_state_path(tmp_output_dir)
    assert resolved == str(Path(tmp_output_dir) / 'time_accounting.json')


def test_reconcile_resolves_to_run_nightly_state(sample_allocations_path,
                                                 tmp_output_dir):
    """run-nightly charges 3.5h; reconcile (no explicit --state) against the
    SAME path with actual == scheduled gives net delta ~ 0 — not a double charge.
    """
    state_path = _default_state_path(tmp_output_dir)

    # Simulate the run-nightly charge.
    acct = TimeAccountant.from_yaml(sample_allocations_path, state_path=state_path)
    acct.charge('MAGNETS-Stubbs', 3.5, 'grey', date='2026-10-15')
    used_after_schedule = acct.get_remaining('MAGNETS-Stubbs', 'grey')

    assert Path(state_path).exists()

    # Reconcile resolving to the SAME file (as cmd_reconcile would when --state
    # is None: _default_state_path(output_dir)).
    acct2 = TimeAccountant.from_yaml(sample_allocations_path, state_path=state_path)
    delta = acct2.reconcile(
        program='MAGNETS-Stubbs', actual_hours=3.5,
        moon_phase='grey', date='2026-10-15',
    )

    assert abs(delta) < 0.001  # actual == scheduled → no adjustment
    # Remaining unchanged: not double-charged.
    assert abs(acct2.get_remaining('MAGNETS-Stubbs', 'grey')
               - used_after_schedule) < 0.001


def test_reconcile_against_empty_state_would_double_charge(sample_allocations_path,
                                                           tmp_path):
    """Guard/illustration: reconciling against a FRESH/EMPTY state file (the old
    buggy default) charges delta = actual - 0. The fix is to resolve to the
    night's state file instead (tested above).
    """
    empty_state = str(tmp_path / 'fresh' / 'time_accounting.json')
    acct = TimeAccountant.from_yaml(sample_allocations_path, state_path=empty_state)
    delta = acct.reconcile(
        program='MAGNETS-Stubbs', actual_hours=3.5,
        moon_phase='grey', date='2026-10-15',
    )
    # Demonstrates the failure mode the path-unification fix avoids.
    assert abs(delta - 3.5) < 0.001


def test_reconcile_charges_only_the_delta(sample_allocations_path, tmp_output_dir):
    """actual > scheduled charges the difference, not the full actual."""
    state_path = _default_state_path(tmp_output_dir)
    acct = TimeAccountant.from_yaml(sample_allocations_path, state_path=state_path)
    acct.charge('MAGNETS-Stubbs', 3.5, 'grey', date='2026-10-15')

    acct2 = TimeAccountant.from_yaml(sample_allocations_path, state_path=state_path)
    delta = acct2.reconcile(
        program='MAGNETS-Stubbs', actual_hours=4.0,
        moon_phase='grey', date='2026-10-15',
    )
    assert abs(delta - 0.5) < 0.001
    # Total used in grey should be 4.0, not 3.5 + 4.0.
    used = acct2.allocations['MAGNETS-Stubbs'].used_hours['grey']
    assert abs(used - 4.0) < 0.001


# ---------------------------------------------------------------------------
# W6 / R4 — charge science time, not padded wall-clock
# ---------------------------------------------------------------------------

def test_scheduled_entry_charges_science_not_padding(sample_allocations_path,
                                                     tmp_output_dir):
    """An entry whose wall-clock window was stretched by N minutes charges only
    exposure + overhead (charged_minutes), with padding_minutes recording slack.
    """
    state_path = _default_state_path(tmp_output_dir)
    acct = TimeAccountant.from_yaml(sample_allocations_path, state_path=state_path)

    t0 = Time('2026-10-15 04:00:00', scale='utc')
    # Science: 2 x 900s = 1800s = 30 min exposure + 1 min overhead = 31 min.
    charged = 2 * 900 / 60.0 + 1.0
    # Wall-clock stretched to 50 min → 19 min padding.
    end = t0 + 50 * u.minute
    wall_min = (end - t0).to(u.minute).value
    padding = wall_min - charged

    entry = ScheduledEntry(
        target=Target(name='SN1', program='MAGNETS-Stubbs'),
        start=t0, end=end,
        n_exp=2, exp_sec=900, exp_str='2x900s',
        program='MAGNETS-Stubbs',
        charged_minutes=charged,
        padding_minutes=padding,
    )

    assert abs(entry.charged_minutes - 31.0) < 1e-6
    assert abs(entry.padding_minutes - 19.0) < 1e-6
    # The wall-clock window is larger than what we bill.
    assert wall_min > entry.charged_minutes

    # Charging the science value (as create_schedule now does) bills 31 min,
    # not the 50-min window.
    acct.charge(entry.program, entry.charged_minutes / 60.0, 'grey',
                date='2026-10-15')
    used = acct.allocations['MAGNETS-Stubbs'].used_hours['grey']
    assert abs(used - charged / 60.0) < 1e-6
    assert used < wall_min / 60.0
