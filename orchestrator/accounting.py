"""Time accounting for MAGNETS multi-program observing allocations.

Tracks per-program time budgets across moon phases, supports
charge-on-schedule and post-night reconciliation.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .models import ProgramAllocation

logger = logging.getLogger(__name__)


@dataclass
class TimeAccountant:
    """Track and manage observing time across MAGNETS programs."""

    allocations: dict = field(default_factory=dict)  # program -> ProgramAllocation
    semester: str = ''
    default_program: str = 'default'
    state_path: str = 'time_accounting.json'
    charge_log: list = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str,
                  state_path: str = 'time_accounting.json') -> 'TimeAccountant':
        """Load allocations from YAML, merge with existing state if present.

        Parameters
        ----------
        yaml_path : str
            Path to allocations YAML file.
        state_path : str
            Path to JSON state file for persisting used_hours and charge log.
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        allocations = {}
        for prog in data.get('programs', []):
            alloc = ProgramAllocation(
                program=prog['program'],
                pi=prog.get('pi', ''),
                semester=data.get('semester', ''),
                allocated_hours=prog.get('allocated_hours', {
                    'dark': 0.0, 'grey': 0.0, 'bright': 0.0,
                }),
                used_hours={'dark': 0.0, 'grey': 0.0, 'bright': 0.0},
                phase_preference=prog.get('phase_preference', 'peak'),
                ranking_profile=prog.get('ranking_profile', 'ia'),
            )
            allocations[alloc.program] = alloc

        accountant = cls(
            allocations=allocations,
            semester=data.get('semester', ''),
            default_program=data.get('default_program', 'default'),
            state_path=state_path,
        )

        # Merge existing state if present
        state_file = Path(state_path)
        if state_file.exists():
            accountant._load_state(state_file)

        return accountant

    def _load_state(self, state_file: Path) -> None:
        """Merge persisted used_hours and charge_log into current allocations."""
        with open(state_file) as f:
            state = json.load(f)

        for prog_name, prog_state in state.get('programs', {}).items():
            if prog_name in self.allocations:
                used = prog_state.get('used_hours', {})
                for phase in ('dark', 'grey', 'bright'):
                    self.allocations[prog_name].used_hours[phase] = used.get(
                        phase, 0.0)

        self.charge_log = state.get('charge_log', [])
        logger.info("Loaded existing state from %s (%d log entries)",
                    state_file, len(self.charge_log))

    def charge(self, program: str, hours: float, moon_phase: str,
               date: str = '') -> None:
        """Charge scheduled hours against a program's budget.

        Parameters
        ----------
        program : str
            Program name (must exist in allocations).
        hours : float
            Hours to charge.
        moon_phase : str
            One of 'dark', 'grey', 'bright'.
        date : str
            Observing date for the charge log.
        """
        if program not in self.allocations:
            logger.warning("Unknown program '%s'; charging to '%s'",
                           program, self.default_program)
            program = self.default_program
            if program not in self.allocations:
                logger.error("Default program '%s' not in allocations", program)
                return

        alloc = self.allocations[program]
        alloc.used_hours[moon_phase] = alloc.used_hours.get(moon_phase, 0.0) + hours

        self.charge_log.append({
            'date': date,
            'program': program,
            'hours': round(hours, 3),
            'moon_phase': moon_phase,
            'type': 'schedule',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })

        remaining = alloc.allocated_hours.get(moon_phase, 0.0) - alloc.used_hours[moon_phase]
        logger.info("Charged %.2fh %s to %s (%.1fh remaining in %s)",
                    hours, moon_phase, program, remaining, moon_phase)

        self._persist()

    def reconcile(self, program: str, actual_hours: float,
                  moon_phase: str, date: str) -> float:
        """Post-night reconciliation: adjust for actual vs scheduled time.

        Returns the delta (actual - previously charged for this date/program).
        """
        if program not in self.allocations:
            logger.error("Cannot reconcile unknown program '%s'", program)
            return 0.0

        # Sum charges for this program/date
        scheduled = sum(
            e['hours'] for e in self.charge_log
            if e['program'] == program and e['date'] == date
            and e['type'] == 'schedule'
        )
        # Sum prior reconciliations
        prior_reconcile = sum(
            e['hours'] for e in self.charge_log
            if e['program'] == program and e['date'] == date
            and e['type'] == 'reconcile'
        )

        delta = actual_hours - (scheduled + prior_reconcile)
        if abs(delta) < 0.001:
            logger.info("No reconciliation needed for %s on %s", program, date)
            return 0.0

        alloc = self.allocations[program]
        alloc.used_hours[moon_phase] = alloc.used_hours.get(moon_phase, 0.0) + delta

        self.charge_log.append({
            'date': date,
            'program': program,
            'hours': round(delta, 3),
            'moon_phase': moon_phase,
            'type': 'reconcile',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })

        logger.info("Reconciled %s on %s: delta=%.2fh (scheduled=%.2f, actual=%.2f)",
                    program, date, delta, scheduled, actual_hours)

        self._persist()
        return delta

    def get_budget_factor(self, program: str,
                          moon_phase: Optional[str] = None) -> float:
        """Phase-aware budget priority multiplier.

        Scores how much of a program's allocation IN THE RELEVANT MOON PHASE is
        still available, expressed as a fraction of that phase's allocation
        (not a hard hour threshold). A program flush with dark time but out of
        grey time scores low on a grey night.

        Parameters
        ----------
        program : str
            Program name.
        moon_phase : str, optional
            'dark', 'grey', or 'bright'. If omitted, falls back to the total
            remaining across all phases vs total allocation.

        Returns
        -------
        float
            1.0 if >50% of the phase allocation remains, 0.5 if any remains,
            0.1 if exhausted (or the phase allocation is 0).
        """
        if program not in self.allocations:
            return 1.0

        alloc = self.allocations[program]
        if moon_phase:
            allocated = alloc.allocated_hours.get(moon_phase, 0.0)
            remaining = self.get_remaining(program, moon_phase)
        else:
            allocated = sum(alloc.allocated_hours.get(p, 0.0)
                            for p in ('dark', 'grey', 'bright'))
            remaining = alloc.remaining_hours

        # No allocation in this phase → nothing to spend → treat as exhausted.
        if allocated <= 0.0:
            return 0.1

        frac = remaining / allocated
        if frac > 0.5:
            return 1.0
        elif frac > 0.0:
            return 0.5
        return 0.1

    def get_ranking_profile(self, program: str) -> str:
        """The program's configured ranking profile ('ia' default)."""
        if program in self.allocations:
            return getattr(self.allocations[program], 'ranking_profile', 'ia') or 'ia'
        return 'ia'

    def get_phase_preference(self, program: str) -> str:
        """Light-curve phase the program wants its SNe at.

        Returns the program's configured ``phase_preference`` ('peak' or
        'rising'), or 'peak' for an unknown program (the conservative default —
        most MAGNETS science is cosmology/standardization).
        """
        if program not in self.allocations:
            return 'peak'
        return self.allocations[program].phase_preference or 'peak'

    def get_remaining(self, program: str,
                      moon_phase: Optional[str] = None) -> float:
        """Remaining hours for a program, optionally filtered by moon phase."""
        if program not in self.allocations:
            return 0.0

        alloc = self.allocations[program]
        if moon_phase:
            return (alloc.allocated_hours.get(moon_phase, 0.0)
                    - alloc.used_hours.get(moon_phase, 0.0))

        return alloc.remaining_hours

    def summary(self) -> dict:
        """Per-program summary for reporting."""
        result = {}
        for name, alloc in self.allocations.items():
            result[name] = {
                'pi': alloc.pi,
                'allocated': dict(alloc.allocated_hours),
                'used': dict(alloc.used_hours),
                'remaining': {
                    p: alloc.allocated_hours.get(p, 0.0) - alloc.used_hours.get(p, 0.0)
                    for p in ('dark', 'grey', 'bright')
                },
                'total_remaining': alloc.remaining_hours,
                'budget_factor': self.get_budget_factor(name),
            }
        return result

    def _persist(self) -> None:
        """Write current state to JSON."""
        state = {
            'semester': self.semester,
            'programs': {},
            'charge_log': self.charge_log,
        }
        for name, alloc in self.allocations.items():
            state['programs'][name] = {
                'pi': alloc.pi,
                'allocated_hours': dict(alloc.allocated_hours),
                'used_hours': dict(alloc.used_hours),
            }

        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.debug("Persisted state to %s", self.state_path)
