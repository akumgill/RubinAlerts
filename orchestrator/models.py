"""Data models for the LLAMAS observing plan generator."""

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, List

from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

logger = logging.getLogger(__name__)


@dataclass
class Target:
    """Canonical target representation for scheduling."""

    # Required
    name: str = ''
    ra_deg: float = float('nan')
    dec_deg: float = float('nan')

    # Optional with defaults
    priority: int = 3
    mag: float = float('nan')
    mag_filter: str = ''
    redshift: float = float('nan')
    exposure_minutes: float = float('nan')
    moon_constraint: str = 'any'
    notes: str = ''  # free-text, human-readable; NOT scored
    # Structured, validated scheduling tags (controlled vocabulary — see
    # prioritizer.KEYWORD_WEIGHTS). Populated/validated at ingestion
    # (normalize.py); exact membership drives the keyword adjustment, replacing
    # the old substring scan of ``notes``.
    keywords: list = field(default_factory=list)
    # PI override ("non-negotiable"): set True at ingestion by an 'override' /
    # 'mandatory' tag. Forces the target onto the schedule (planner reserves it,
    # bypassing the moon-phase filter but not the physics).
    mandatory: bool = False
    source: str = ''
    program: str = 'default'
    merit_score: float = float('nan')
    phase_weight: float = float('nan')  # w_time from alert pipeline: exp(-dt²/2τ²)
    delta_t: float = float('nan')  # days from peak; <0 rising, 0 peak, >0 declining
    # Per-target integration ledger fields (W11). Populated by run_nightly from
    # the TargetLedger: cumulative integration so far, completeness fraction
    # (cumulative / required_full), and the FULL required exposure in minutes
    # (vs exposure_minutes, which is set to the REMAINING time to observe).
    cumulative_minutes: float = float('nan')
    completeness_fraction: float = float('nan')
    required_minutes_full: float = float('nan')
    # Per-component composite-score breakdown from prioritizer.rank_targets
    # (keys: science, budget, phase, observability, keyword_adj, *_term, total).
    # None until the target has been ranked.
    score_breakdown: Optional[dict] = field(default=None, repr=False)

    # Planner-populated fields
    transit_time: Optional[Time] = field(default=None, repr=False)
    min_airmass: float = float('nan')
    window_start: Optional[Time] = field(default=None, repr=False)
    window_end: Optional[Time] = field(default=None, repr=False)
    window_hours: float = 0.0

    @cached_property
    def coord(self) -> SkyCoord:
        """SkyCoord from ra_deg, dec_deg (cached)."""
        return SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)

    @property
    def ra_hms(self) -> str:
        """RA formatted as HH:MM:SS.ss."""
        hms = self.coord.ra.hms
        return '{:02d}:{:02d}:{:05.2f}'.format(
            int(hms.h), int(abs(hms.m)), abs(hms.s))

    @property
    def dec_dms(self) -> str:
        """Dec formatted as +DD:MM:SS.s."""
        dms = self.coord.dec.dms
        sign = '+' if dms.d >= 0 else '-'
        return '{}{:02d}:{:02d}:{:04.1f}'.format(
            sign, int(abs(dms.d)), int(abs(dms.m)), abs(dms.s))


@dataclass
class ScheduledEntry:
    """A target assigned to a specific time slot in the observing plan."""

    target: Target = field(default_factory=Target)
    start: Optional[Time] = None
    end: Optional[Time] = None
    airmass: float = float('nan')
    exp_str: str = ''
    n_exp: int = 1
    exp_sec: int = 0
    program: str = ''
    # Science time actually integrated = exposure + overhead, in minutes.
    # This is what gets billed to the program (and, in W11, to the per-target
    # ledger) — NOT the wall-clock (end - start), which may be padded by
    # gap-fill / end-of-night stretch for the visual timeline.
    charged_minutes: float = float('nan')
    # Dead-time-avoidance slack folded into the wall-clock window (stretch +
    # gap-fill). Recorded for transparency; never billed.
    padding_minutes: float = 0.0


@dataclass
class ProgramAllocation:
    """Time allocation for a single observing program."""

    program: str = ''
    pi: str = ''
    semester: str = ''
    allocated_hours: dict = field(default_factory=lambda: {
        'dark': 0.0, 'grey': 0.0, 'bright': 0.0,
    })
    used_hours: dict = field(default_factory=lambda: {
        'dark': 0.0, 'grey': 0.0, 'bright': 0.0,
    })
    # Light-curve phase the program wants its SNe at: 'peak' (cosmology/
    # standardization, best S/N at the SALT epoch) or 'rising' (progenitor/CSM/
    # flash spectroscopy, early ejecta). Maps to a preferred delta_t offset via
    # LLAMASConfig.phase_preference_offsets in the prioritizer.
    phase_preference: str = 'peak'
    # Which RankingProfile scores this program's targets ('ia' or 'exotic';
    # see core.magellan_planning.RANKING_PROFILES). Selects the
    # merit_<profile> column from candidates.csv and scopes the P1-P4
    # quartiles to the program's own cohort -- a 50/50 budget split only
    # binds when each program ranks by its own science.
    ranking_profile: str = 'ia'

    @property
    def remaining_hours(self) -> float:
        """Total remaining hours across all moon phases."""
        return sum(
            self.allocated_hours.get(p, 0.0) - self.used_hours.get(p, 0.0)
            for p in ('dark', 'grey', 'bright')
        )


@dataclass
class ObsPlan:
    """Complete observing plan for a single night."""

    date: str = ''
    evening_twilight: Optional[Time] = None
    morning_twilight: Optional[Time] = None
    moon_phase: str = 'grey'

    scheduled: List[ScheduledEntry] = field(default_factory=list)
    backup: List[Target] = field(default_factory=list)
    # Targets excluded from scheduling because the per-target integration ledger
    # marks them as already having sufficient cumulative integration (W11).
    completed: List[Target] = field(default_factory=list)
    # Mandatory (PI-override) targets that could NOT be scheduled because they
    # never reach the airmass limit tonight (physics, not policy). Surfaced in
    # the summary so a non-negotiable target never silently vanishes.
    unschedulable_mandatory: List[Target] = field(default_factory=list)

    standards_start: Optional[dict] = None
    standards_end: Optional[dict] = None
    # Mid-night standards inserted on a fixed cadence for long nights (R10).
    # Each dict mirrors standards_start/end plus a 'time' key (astropy Time)
    # marking when in the night it should be observed. Empty for short nights.
    standards_mid: List[dict] = field(default_factory=list)

    # Which scoring path produced the ranking (R18): 'prioritizer' when
    # composite scores were supplied, else 'fallback/priority-only'.
    scoring_mode: str = 'fallback/priority-only'
    # {target.name: breakdown_dict} from the prioritizer (R14). Empty when the
    # fallback path ran. Used by write_summary to persist score_breakdown.json
    # and render the per-target breakdown table.
    score_breakdowns: dict = field(default_factory=dict)
    # Multi-group alerts (W12): objects wanted by >1 distinct MAGNETS program.
    # Each dict: {name, ra_deg, dec_deg, programs, phase_preferences,
    # observed_phase, same_phase}. Written to multi_group_alerts.json and
    # rendered in the summary so coordinating PIs can see the overlap.
    multi_group_alerts: list = field(default_factory=list)

    @property
    def night_duration_hours(self) -> float:
        """Total dark time in hours."""
        if self.evening_twilight is None or self.morning_twilight is None:
            return 0.0
        return (self.morning_twilight - self.evening_twilight).to(u.hour).value

    @property
    def scheduled_minutes(self) -> float:
        """Total scheduled observation time in minutes."""
        total = 0.0
        for entry in self.scheduled:
            if entry.start is not None and entry.end is not None:
                total += (entry.end - entry.start).to(u.minute).value
        return total

    @property
    def efficiency(self) -> float:
        """Fraction of night used for scheduled observations."""
        night_min = self.night_duration_hours * 60.0
        if night_min <= 0:
            return 0.0
        return self.scheduled_minutes / night_min
