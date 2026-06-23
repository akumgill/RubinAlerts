"""Per-target cumulative integration ledger for LLAMAS follow-up.

Tracks how much science integration time a target has accumulated ACROSS
nights, keyed by sky coordinate (so a target renamed between nights — e.g. an
internal alert id later upgraded to a TNS/IAU name — is still recognised as the
same object). This is deliberately SEPARATE from the per-PROGRAM hour budgets
in ``accounting.py``: the accountant answers "how much of PI X's allocation is
left?", while this ledger answers "does THIS target already have enough
integration time that we should stop re-observing it?".

The ledger feeds two scheduling decisions:

* a multiplicative ``completeness_factor`` folded into the composite science
  core (so a finished target falls to the bottom of the ranking); and
* a per-night ``remaining_minutes``, so a partially-integrated target is
  scheduled only for the time it still needs.

State persists as JSON, mirroring ``TimeAccountant``'s load/_persist pattern.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from astropy.coordinates import SkyCoord
import astropy.units as u

from .models import Target

logger = logging.getLogger(__name__)


@dataclass
class TargetLedgerEntry:
    """Cumulative integration record for a single sky position."""

    coord_key: str = ''
    canonical_name: str = ''
    aliases: list = field(default_factory=list)
    ra_deg: float = float('nan')
    dec_deg: float = float('nan')
    cumulative_science_seconds: float = 0.0
    # Per-night required-exposure snapshots: {date, required_s, mag, redshift}.
    required_seconds_history: list = field(default_factory=list)
    # Dates on which any science time was charged.
    nights_observed: list = field(default_factory=list)
    # Audit trail: {date, seconds, type ('schedule'|'reconcile'), timestamp}.
    charge_log: list = field(default_factory=list)


@dataclass
class TargetLedger:
    """Coordinate-keyed cumulative integration ledger across nights."""

    entries: dict = field(default_factory=dict)  # coord_key -> TargetLedgerEntry
    state_path: str = 'target_ledger.json'
    # Two targets within this angular separation are treated as the same object.
    match_radius_arcsec: float = 2.0
    # A target is "done" once it reaches this fraction of its required time.
    satisfied_fraction: float = 0.95
    # ...or once the time it still needs is shorter than a worthwhile block.
    min_block_minutes: float = 15.0
    # Floor on the completeness factor so a near-done high-priority target is
    # nudged down but never fully zeroed while it still needs real time.
    min_factor: float = 0.15

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, state_path: str = 'target_ledger.json') -> 'TargetLedger':
        """Load a ledger from JSON, or return an empty one if absent.

        Missing file is NOT an error (mirrors TimeAccountant.from_yaml's
        merge-if-present pattern): a first-ever night starts from scratch.
        """
        ledger = cls(state_path=state_path)
        state_file = Path(state_path)
        if state_file.exists():
            ledger._load_state(state_file)
        else:
            logger.info("No existing target ledger at %s; starting empty",
                        state_path)
        return ledger

    def _load_state(self, state_file: Path) -> None:
        """Read persisted entries into ``self.entries``."""
        with open(state_file) as f:
            state = json.load(f)

        for key, ent in state.get('entries', {}).items():
            self.entries[key] = TargetLedgerEntry(
                coord_key=ent.get('coord_key', key),
                canonical_name=ent.get('canonical_name', ''),
                aliases=list(ent.get('aliases', [])),
                ra_deg=ent.get('ra_deg', float('nan')),
                dec_deg=ent.get('dec_deg', float('nan')),
                cumulative_science_seconds=ent.get(
                    'cumulative_science_seconds', 0.0),
                required_seconds_history=list(
                    ent.get('required_seconds_history', [])),
                nights_observed=list(ent.get('nights_observed', [])),
                charge_log=list(ent.get('charge_log', [])),
            )
        logger.info("Loaded target ledger from %s (%d entries)",
                    state_file, len(self.entries))

    def _persist(self) -> None:
        """Write current entries to JSON (mirrors TimeAccountant._persist)."""
        state = {'entries': {k: asdict(v) for k, v in self.entries.items()}}
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)
        logger.debug("Persisted target ledger to %s", self.state_path)

    # ------------------------------------------------------------------
    # Coordinate matching / entry lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _coord_key(ra_deg: float, dec_deg: float) -> str:
        """Deterministic string key from coordinates (5 dp ~ 0.04\")."""
        return f"{ra_deg:.5f}_{dec_deg:+.5f}"

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalise a target name for dedup (strip, collapse spaces)."""
        return ' '.join((name or '').split()).strip()

    @staticmethod
    def _is_tns_name(name: str) -> bool:
        """True if the name looks like a TNS/IAU designation (SN.../AT...)."""
        n = (name or '').strip().upper()
        return n.startswith('SN') or n.startswith('AT')

    def _find_entry(self, target: Target) -> Optional[TargetLedgerEntry]:
        """Match an existing entry by angular separation (not exact key).

        Returns the closest entry within ``match_radius_arcsec`` or None.
        """
        coord = SkyCoord(ra=target.ra_deg * u.deg, dec=target.dec_deg * u.deg)
        best = None
        best_sep = self.match_radius_arcsec
        for ent in self.entries.values():
            ent_coord = SkyCoord(ra=ent.ra_deg * u.deg, dec=ent.dec_deg * u.deg)
            sep = coord.separation(ent_coord).arcsec
            if sep <= best_sep:
                best_sep = sep
                best = ent
        return best

    def get_or_create(self, target: Target) -> TargetLedgerEntry:
        """Find the entry for this target by coordinate, or create one.

        On a coordinate match with a previously-unseen name, the name is added
        to ``aliases`` (normalised, deduped) and the canonical name is upgraded
        if the new name looks like a TNS/IAU designation.
        """
        new_name = self._normalize_name(target.name)
        ent = self._find_entry(target)

        if ent is None:
            key = self._coord_key(target.ra_deg, target.dec_deg)
            ent = TargetLedgerEntry(
                coord_key=key,
                canonical_name=new_name,
                aliases=[new_name] if new_name else [],
                ra_deg=target.ra_deg,
                dec_deg=target.dec_deg,
            )
            self.entries[key] = ent
            logger.debug("New ledger entry %s (%s)", key, new_name)
            return ent

        # Matched an existing entry: reconcile names.
        if new_name and new_name not in ent.aliases:
            ent.aliases.append(new_name)
            logger.debug("Added alias '%s' to ledger entry %s",
                         new_name, ent.coord_key)
        # Upgrade canonical name if the new name is a TNS/IAU designation and
        # the current canonical isn't.
        if (new_name and self._is_tns_name(new_name)
                and not self._is_tns_name(ent.canonical_name)):
            logger.debug("Upgraded canonical name %s -> %s",
                         ent.canonical_name, new_name)
            ent.canonical_name = new_name
        return ent

    # ------------------------------------------------------------------
    # Completeness queries
    # ------------------------------------------------------------------

    def cumulative_seconds(self, target: Target) -> float:
        """Cumulative science seconds integrated on this target so far."""
        ent = self._find_entry(target)
        return ent.cumulative_science_seconds if ent else 0.0

    def completeness_fraction(self, target: Target,
                              required_minutes: float) -> float:
        """Fraction of the required integration already accumulated."""
        if not required_minutes or required_minutes <= 0:
            return 0.0
        return self.cumulative_seconds(target) / (required_minutes * 60.0)

    def remaining_minutes(self, target: Target,
                          required_minutes: float) -> float:
        """Integration minutes still needed (>= 0)."""
        return max(0.0, required_minutes
                   - self.cumulative_seconds(target) / 60.0)

    def is_satisfied(self, target: Target, required_minutes: float) -> bool:
        """True if the target has enough integration to skip re-observing.

        Satisfied when either the completeness fraction has reached
        ``satisfied_fraction`` OR the remaining time is shorter than a
        worthwhile observing block (``min_block_minutes``).
        """
        frac = self.completeness_fraction(target, required_minutes)
        if frac >= self.satisfied_fraction:
            return True
        return self.remaining_minutes(target, required_minutes) < self.min_block_minutes

    def completeness_factor(self, target: Target,
                            required_minutes: float) -> float:
        """Multiplicative score factor in (0, 1] from completeness.

        Fully done (>=1.0) -> 0.0 (drop to the bottom); nothing done (<=0) ->
        1.0 (full weight); in between -> 1 - fraction, floored at
        ``min_factor`` so a near-done high-priority target is still rankable.
        """
        frac = self.completeness_fraction(target, required_minutes)
        if frac >= 1.0:
            return 0.0
        if frac <= 0.0:
            return 1.0
        return max(self.min_factor, 1.0 - frac)

    # ------------------------------------------------------------------
    # Charging / reconciliation
    # ------------------------------------------------------------------

    def charge(self, target: Target, science_seconds: float, date: str,
               mag: float = float('nan'), redshift: float = float('nan'),
               required_seconds: float = float('nan')) -> None:
        """Charge science integration time against a target's ledger entry."""
        ent = self.get_or_create(target)
        ent.cumulative_science_seconds += science_seconds
        ent.required_seconds_history.append({
            'date': date,
            'required_s': round(required_seconds, 1)
            if required_seconds == required_seconds else None,  # NaN -> None
            'mag': mag if mag == mag else None,
            'redshift': redshift if redshift == redshift else None,
        })
        if date not in ent.nights_observed:
            ent.nights_observed.append(date)
        ent.charge_log.append({
            'date': date,
            'seconds': round(science_seconds, 1),
            'type': 'schedule',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Charged %.0fs to %s (cumulative %.0fs)",
                    science_seconds, ent.canonical_name,
                    ent.cumulative_science_seconds)
        self._persist()

    def reconcile(self, target: Target, actual_seconds: float,
                  date: str) -> float:
        """Adjust cumulative so the date's total matches ``actual_seconds``.

        Computes the delta versus all prior schedule+reconcile charges for
        that date, applies it, logs a 'reconcile' entry, and persists. Returns
        the delta (0.0 if no change was needed).
        """
        ent = self.get_or_create(target)
        prior = sum(e['seconds'] for e in ent.charge_log
                    if e['date'] == date
                    and e['type'] in ('schedule', 'reconcile'))
        delta = actual_seconds - prior
        if abs(delta) < 0.1:
            logger.info("No target reconciliation needed for %s on %s",
                        ent.canonical_name, date)
            return 0.0

        ent.cumulative_science_seconds += delta
        ent.charge_log.append({
            'date': date,
            'seconds': round(delta, 1),
            'type': 'reconcile',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Reconciled %s on %s: delta=%.0fs (prior=%.0f, actual=%.0f)",
                    ent.canonical_name, date, delta, prior, actual_seconds)
        self._persist()
        return delta

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Per-entry summary keyed by canonical name."""
        result = {}
        for ent in self.entries.values():
            cumulative_min = ent.cumulative_science_seconds / 60.0
            required_min = float('nan')
            if ent.required_seconds_history:
                latest = ent.required_seconds_history[-1].get('required_s')
                if latest is not None:
                    required_min = latest / 60.0
            if required_min and required_min == required_min and required_min > 0:
                fraction = cumulative_min / required_min
            else:
                fraction = float('nan')
            if fraction == fraction and fraction >= self.satisfied_fraction:
                status = 'satisfied'
            elif cumulative_min > 0:
                status = 'partial'
            else:
                status = 'pending'
            result[ent.canonical_name] = {
                'name': ent.canonical_name,
                'cumulative_min': cumulative_min,
                'required_min': required_min,
                'fraction': fraction,
                'status': status,
            }
        return result
