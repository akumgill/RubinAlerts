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
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from astropy.coordinates import SkyCoord
import astropy.units as u

from .models import Target

logger = logging.getLogger(__name__)


def phase_bucket(delta_t, window: float) -> str:
    """Light-curve phase bucket for a signed time-from-peak (days).

    Buckets so integration time can be tracked SEPARATELY per phase — a target
    "done at peak" should not have its still-needed RISING time suppressed.

        NaN / None    -> 'all'        (no timing info; one undifferentiated pool)
        delta_t < -W  -> 'rising'     (pre-peak, early ejecta)
        |delta_t| <= W -> 'peak'      (within the SALT epoch window)
        delta_t > W   -> 'declining'  (post-peak)

    where W = ``window`` (config.phase_bucket_window_days).
    """
    if delta_t is None:
        return 'all'
    try:
        dt = float(delta_t)
    except (TypeError, ValueError):
        return 'all'
    if not math.isfinite(dt):
        return 'all'
    if dt < -window:
        return 'rising'
    if dt > window:
        return 'declining'
    return 'peak'


@dataclass
class TargetLedgerEntry:
    """Cumulative integration record for a single sky position.

    Integration is tracked SEPARATELY per phase bucket
    (``cumulative_seconds_by_phase``) so a target's peak-epoch time and its
    rising-epoch time do not cross-satisfy. ``cumulative_science_seconds`` is a
    read accessor summing across buckets, preserving the prior scalar API.
    """

    coord_key: str = ''
    canonical_name: str = ''
    aliases: list = field(default_factory=list)
    ra_deg: float = float('nan')
    dec_deg: float = float('nan')
    # Per-phase cumulative science seconds: {'all'|'rising'|'peak'|'declining'
    # -> seconds}. Old-format scalars migrate into the 'all' bucket on load.
    cumulative_seconds_by_phase: dict = field(default_factory=dict)
    # Distinct programs that have charged this target (for multi-group alerts).
    programs: list = field(default_factory=list)
    # Per-night required-exposure snapshots: {date, required_s, mag, redshift}.
    required_seconds_history: list = field(default_factory=list)
    # Dates on which any science time was charged.
    nights_observed: list = field(default_factory=list)
    # Audit trail: {date, seconds, phase, program, type, timestamp}.
    charge_log: list = field(default_factory=list)

    @property
    def cumulative_science_seconds(self) -> float:
        """Total science seconds across all phase buckets (scalar-API accessor)."""
        return sum(self.cumulative_seconds_by_phase.values())


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
            # Per-phase buckets. MIGRATE old-format state: an entry persisted
            # before phase-splitting carries a scalar cumulative_science_seconds
            # -> load it into the undifferentiated 'all' bucket.
            by_phase = dict(ent.get('cumulative_seconds_by_phase', {}))
            if not by_phase and 'cumulative_science_seconds' in ent:
                old_scalar = ent.get('cumulative_science_seconds', 0.0) or 0.0
                if old_scalar:
                    by_phase = {'all': old_scalar}
            self.entries[key] = TargetLedgerEntry(
                coord_key=ent.get('coord_key', key),
                canonical_name=ent.get('canonical_name', ''),
                aliases=list(ent.get('aliases', [])),
                ra_deg=ent.get('ra_deg', float('nan')),
                dec_deg=ent.get('dec_deg', float('nan')),
                cumulative_seconds_by_phase=by_phase,
                programs=list(ent.get('programs', [])),
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

    def cumulative_seconds(self, target: Target, phase=None) -> float:
        """Cumulative science seconds integrated on this target so far.

        ``phase`` given -> seconds in that bucket only (0.0 if absent); None ->
        the SUM across all buckets (the prior scalar behaviour).
        """
        ent = self._find_entry(target)
        if ent is None:
            return 0.0
        if phase is None:
            return ent.cumulative_science_seconds
        return ent.cumulative_seconds_by_phase.get(phase, 0.0)

    def completeness_fraction(self, target: Target,
                              required_minutes: float, phase=None) -> float:
        """Fraction of the required integration already accumulated.

        With ``phase`` set, the fraction is judged against that bucket only, so
        a target satisfied at peak can still be incomplete in the rising bucket.
        """
        if not required_minutes or required_minutes <= 0:
            return 0.0
        return self.cumulative_seconds(target, phase) / (required_minutes * 60.0)

    def remaining_minutes(self, target: Target,
                          required_minutes: float, phase=None) -> float:
        """Integration minutes still needed (>= 0), optionally per phase."""
        return max(0.0, required_minutes
                   - self.cumulative_seconds(target, phase) / 60.0)

    def is_satisfied(self, target: Target, required_minutes: float,
                     phase=None) -> bool:
        """True if the target has enough integration to skip re-observing.

        Satisfied when either the completeness fraction has reached
        ``satisfied_fraction`` OR the remaining time is shorter than a
        worthwhile observing block (``min_block_minutes``). With ``phase`` set,
        the judgement is per-bucket: rising-phase integration does not satisfy
        the peak bucket and vice versa.
        """
        frac = self.completeness_fraction(target, required_minutes, phase)
        if frac >= self.satisfied_fraction:
            return True
        # The below-block rule exists so we don't chase a 5-minute remainder
        # on a nearly-done target. It must NOT fire on a target with ZERO
        # integration whose TOTAL need is small — a bright target requiring
        # <15 min was born "satisfied" and could never be observed at all
        # (caught live by the 2026-07-14 manual-enqueue demo: SN 2026roc,
        # mag 17.2, excluded at 0/10 min).
        if frac <= 0.0:
            return False
        return self.remaining_minutes(target, required_minutes, phase) \
            < self.min_block_minutes

    def completeness_factor(self, target: Target,
                            required_minutes: float, phase=None) -> float:
        """Multiplicative score factor in (0, 1] from completeness.

        Fully done (>=1.0) -> 0.0 (drop to the bottom); nothing done (<=0) ->
        1.0 (full weight); in between -> 1 - fraction, floored at
        ``min_factor`` so a near-done high-priority target is still rankable.
        """
        frac = self.completeness_fraction(target, required_minutes, phase)
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
               required_seconds: float = float('nan'),
               phase=None, program=None) -> None:
        """Charge science integration time against a target's ledger entry.

        ``phase`` selects the bucket (None -> the undifferentiated 'all' pool);
        ``program`` records which program spent the time (for multi-group
        alerts).
        """
        ent = self.get_or_create(target)
        bucket = phase or 'all'
        ent.cumulative_seconds_by_phase[bucket] = \
            ent.cumulative_seconds_by_phase.get(bucket, 0.0) + science_seconds
        if program and program not in ent.programs:
            ent.programs.append(program)
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
            'phase': bucket,
            'program': program,
            'type': 'schedule',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Charged %.0fs to %s [%s] (cumulative %.0fs)",
                    science_seconds, ent.canonical_name, bucket,
                    ent.cumulative_science_seconds)
        self._persist()

    def reconcile(self, target: Target, actual_seconds: float,
                  date: str, phase=None) -> float:
        """Adjust cumulative so the date's total matches ``actual_seconds``.

        Computes the delta versus all prior schedule+reconcile charges for
        that date, applies it, logs a 'reconcile' entry, and persists. Returns
        the delta (0.0 if no change was needed).

        ``phase`` selects which bucket the true-up adjusts (None -> the 'all'
        bucket, preserving the prior single-pool behaviour). A per-phase
        reconciliation only compares against prior charges in that same bucket.
        """
        ent = self.get_or_create(target)
        bucket = phase or 'all'
        prior = sum(e['seconds'] for e in ent.charge_log
                    if e['date'] == date
                    and e['type'] in ('schedule', 'reconcile')
                    and (phase is None or e.get('phase', 'all') == bucket))
        delta = actual_seconds - prior
        if abs(delta) < 0.1:
            logger.info("No target reconciliation needed for %s on %s",
                        ent.canonical_name, date)
            return 0.0

        ent.cumulative_seconds_by_phase[bucket] = \
            ent.cumulative_seconds_by_phase.get(bucket, 0.0) + delta
        ent.charge_log.append({
            'date': date,
            'seconds': round(delta, 1),
            'phase': bucket,
            'program': None,
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
        """Per-entry summary keyed by canonical name.

        Each entry includes ``cumulative_min_by_phase`` (the per-phase
        breakdown) and the ``programs`` list that charged the target.
        """
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
            by_phase_min = {
                ph: secs / 60.0
                for ph, secs in ent.cumulative_seconds_by_phase.items()
            }
            result[ent.canonical_name] = {
                'name': ent.canonical_name,
                'cumulative_min': cumulative_min,
                'cumulative_min_by_phase': by_phase_min,
                'programs': list(ent.programs),
                'required_min': required_min,
                'fraction': fraction,
                'status': status,
            }
        return result

    def multi_program_entries(self) -> list:
        """Ledger entries charged by more than one distinct program.

        These are objects multiple MAGNETS programs have integrated on — useful
        for flagging shared targets whose phase preferences may conflict.
        """
        return [ent for ent in self.entries.values()
                if len(set(ent.programs)) > 1]
