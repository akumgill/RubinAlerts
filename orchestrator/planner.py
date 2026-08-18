"""Scheduling engine for LLAMAS observing plans.

Computes twilight, observability windows, and creates greedy schedules
optimized for priority and airmass.
"""

import logging
import math
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from astropy.coordinates import SkyCoord, AltAz, get_sun
from astropy.time import Time
import astropy.units as u

from .config import LLAMAS_CONFIG, LLAMASConfig
from .models import Target, ScheduledEntry, ObsPlan
from .target_ledger import phase_bucket

logger = logging.getLogger(__name__)

# Default path to spectrophotometric standards catalog
DEFAULT_STANDARDS_PATH = (
    Path(__file__).parent.parent / 'ref' / 'LDSS_ObsPlan_Generator' / 'standards.txt'
)

# Per-degree penalty applied to the slew (angular separation) from the
# previously-scheduled target. The 1-minute IFU overhead (config.overhead_min)
# assumes negligible slew, so this term compensates by gently discouraging picks
# that cross the sky. It is deliberately *modest* (0.5 pt/deg) so it tie-breaks
# between near-equal candidates and keeps DDF clusters together, rather than
# dominating the science/airmass terms (airmass alone is weighted ×10). This is
# NOT a full path optimizer — just a greedy nudge toward the current pointing.
SLEW_PENALTY = 0.5

# Value-density nudge: a greedy scheduler that ranks purely by score packs
# fewer total-merit minutes than one that (gently) prefers shorter exposures —
# classic knapsack. The bonus is bounded (<= 15 points, less than one
# priority tier) so it reorders near-ties but cannot leapfrog science
# priority: score += BONUS * min(REF/exposure, 3).
EXPOSURE_DENSITY_BONUS = 5.0
EXPOSURE_DENSITY_REF_MIN = 45.0

# Within-night fairness nudge (PI policy, 2026-07-13): a multi-program split
# should stay "at least close" to the allocation shares without being a hard
# wall — the most interesting target can still win, and reconcile trues up
# afterwards. The under-served program (relative to its share of tonight's
# allocations) gets a bounded boost proportional to its deficit; the cap is
# below one priority tier so fairness reorders near-ties, never science.
NIGHT_BALANCE_BONUS = 25.0


# ---------------------------------------------------------------------------
# Twilight
# ---------------------------------------------------------------------------

def calculate_twilight(date_str: str,
                       config: LLAMASConfig = None) -> Tuple[Time, Time]:
    """Compute astronomical twilight times for a given UT date.

    Finds when the Sun crosses the configured sun altitude threshold
    (default -18 deg for astronomical twilight).

    Parameters
    ----------
    date_str : str
        Observing date in YYYY-MM-DD format.
    config : LLAMASConfig, optional
        Configuration; defaults to LLAMAS_CONFIG.

    Returns
    -------
    (evening_twilight, morning_twilight) : tuple of Time
    """
    if config is None:
        config = LLAMAS_CONFIG

    # Search from local noon (UT ~16:00 for Chile) over 24 hours
    noon = Time(f'{date_str} 16:00:00', scale='utc')
    times = noon + np.linspace(0, 24, 1441) * u.hour
    sun_alt = get_sun(times).transform_to(
        AltAz(obstime=times, location=config.location)
    ).alt.deg

    threshold = config.twilight_sun_alt
    evening_idx = None
    morning_idx = None

    for i in range(len(sun_alt) - 1):
        if sun_alt[i] > threshold and sun_alt[i + 1] <= threshold:
            evening_idx = i
        if sun_alt[i] <= threshold and sun_alt[i + 1] > threshold and evening_idx is not None:
            morning_idx = i
            break

    if evening_idx is None or morning_idx is None:
        raise ValueError(f"Could not compute twilight for {date_str}")

    evening = times[evening_idx]
    morning = times[morning_idx]
    logger.info("Twilight: evening %s, morning %s (%.1f hrs)",
                evening.iso[11:16], morning.iso[11:16],
                (morning - evening).to(u.hour).value)
    return evening, morning


# ---------------------------------------------------------------------------
# Airmass utilities
# ---------------------------------------------------------------------------

def _get_airmass(coord: SkyCoord, time: Time, location) -> float:
    """Compute airmass for a single coordinate at a single time."""
    alt = coord.transform_to(AltAz(obstime=time, location=location)).alt.deg
    if alt > 0:
        return 1.0 / np.sin(np.radians(alt))
    return float('inf')


def _get_airmass_grid(coord: SkyCoord, times, location) -> np.ndarray:
    """Compute airmass over a time grid (vectorized)."""
    alt = coord.transform_to(AltAz(obstime=times, location=location)).alt.deg
    with np.errstate(divide='ignore', invalid='ignore'):
        am = 1.0 / np.sin(np.radians(alt))
        am[alt <= 0] = np.inf
    return am


def _airmass_bounds(t, config) -> tuple:
    """Effective (lo, hi) airmass bounds for scheduling ``t`` (stamped #5).

    No per-target range -> (1.0, config.max_airmass), the historical behavior.
    An explicit range is a HARD constraint that OVERRIDES the global limit
    (standards bins may sit entirely above config.max_airmass); a min-only
    range leaves the top unbounded (horizon physics still applies)."""
    lo = getattr(t, 'airmass_min', float('nan'))
    hi = getattr(t, 'airmass_max', float('nan'))
    has_lo = isinstance(lo, (int, float)) and math.isfinite(lo)
    has_hi = isinstance(hi, (int, float)) and math.isfinite(hi)
    if not has_lo and not has_hi:
        return 1.0, config.max_airmass
    return ((float(lo) if has_lo else 1.0),
            (float(hi) if has_hi else float('inf')))


def _am_ok(t, am: float, config) -> bool:
    """Slot-time airmass check against the target's effective bounds. This is
    what enforces a min-airmass band through its mid-window violation (the
    [start, end] envelope from _find_window is not gap-free)."""
    lo, hi = _airmass_bounds(t, config)
    return lo <= am <= hi


def _find_window(coord: SkyCoord, evening: Time, morning: Time,
                 location, max_airmass: float,
                 min_airmass: float = 1.0) -> tuple:
    """Find the observable window for a target.

    ``min_airmass``/``max_airmass`` bound the target's ALLOWED airmass band
    (per-target ranges from stamped #5; the default band is [1, global max]).
    Note the returned [start, end] span is the ENVELOPE of allowed times —
    for a min-airmass band it can contain a mid-night violation (the target
    culminating too high), which the scheduler re-checks per slot.

    Returns
    -------
    (transit_time, min_airmass, window_start, window_end) — transit/min are
    the best allowed point — or (None, None, None, None) if the target never
    enters the allowed band.
    """
    duration_min = (morning - evening).to(u.minute).value
    times = evening + np.linspace(0, duration_min, 500) * u.minute
    am = _get_airmass_grid(coord, times, location)

    obs_mask = (am <= max_airmass) & (am >= min_airmass)
    if not np.any(obs_mask):
        return None, None, None, None

    idx = np.where(obs_mask)[0]
    best = idx[np.argmin(am[idx])]      # best ALLOWED point, not global min
    transit_time = times[best]
    min_am = float(am[best])
    window_start = times[idx[0]]
    window_end = times[idx[-1]]

    return transit_time, min_am, window_start, window_end


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

def _is_honored_mustsee(t: Target, primary_program: Optional[str]) -> bool:
    """True if ``t`` is a must-see (mandatory) target whose guarantee holds
    tonight under the night-primary policy.

    A mandatory target is HONORED when there is no designated primary
    (``primary_program is None`` — backward-compatible: all must-see honored) or
    when its program matches tonight's primary. Non-primary must-see targets are
    NOT honored: they are demoted to normal targets (the gating itself lives in
    ``create_schedule``; this helper just identifies the honored ones so the
    short-window drop can exempt them)."""
    if not getattr(t, 'mandatory', False):
        return False
    return primary_program is None or t.program == primary_program


def compute_observability(targets: List[Target], evening: Time,
                          morning: Time,
                          config: LLAMASConfig = None,
                          primary_program: Optional[str] = None) -> List[Target]:
    """Populate observability windows and filter unobservable targets.

    Parameters
    ----------
    targets : list of Target
        Input targets with ra_deg, dec_deg, exposure_minutes populated.
    evening, morning : Time
        Twilight boundaries.
    config : LLAMASConfig, optional
    primary_program : str, optional
        Tonight's designated primary program (from the nights CSV). HONORED
        must-see targets (mandatory AND allowed by the primary rule) are exempt
        from the short-window drop below so their guarantee actually holds — see
        ``_is_honored_mustsee``. None (the default) honors all must-see targets.

    Returns
    -------
    list of Target
        Observable targets with window fields populated.
    """
    if config is None:
        config = LLAMAS_CONFIG

    observable = []
    for t in targets:
        am_lo, am_hi = _airmass_bounds(t, config)
        transit, min_am, ws, we = _find_window(
            t.coord, evening, morning, config.location,
            max_airmass=am_hi, min_airmass=am_lo,
        )
        if transit is None:
            # No window at all (never reaches the airmass limit). Even an
            # HONORED must-see target cannot be scheduled here — it will land in
            # unschedulable_mandatory + warn in create_schedule. Drop it so it
            # is not carried forward as observable.
            logger.debug("Not observable: %s", t.name)
            continue

        t.transit_time = transit
        t.min_airmass = min_am
        t.window_start = ws
        t.window_end = we
        t.window_hours = (we - ws).to(u.hour).value

        # Check window is long enough for exposure + overhead. EXEMPT honored
        # must-see targets: a guaranteed target with a short-but-nonzero window
        # must still reach the reservation pass (its guarantee outranks the
        # normal minimum-window cutoff), so we keep it even when the window is
        # shorter than a full exposure+overhead block.
        needed = t.exposure_minutes + config.overhead_minutes
        if not math.isfinite(needed):
            needed = config.fallback_exposure_minutes + config.overhead_minutes
        window_min = t.window_hours * 60.0
        if window_min < needed:
            if _is_honored_mustsee(t, primary_program):
                logger.info("Window too short for must-see %s (%.0f < %.0f "
                            "min) — kept (guaranteed)", t.name, window_min,
                            needed)
            else:
                logger.debug("Window too short for %s (%.0f < %.0f min)",
                             t.name, window_min, needed)
                continue

        observable.append(t)

    logger.info("Observable: %d / %d targets", len(observable), len(targets))
    return observable


# ---------------------------------------------------------------------------
# Standard star selection
# ---------------------------------------------------------------------------

def _parse_standards(filename: str) -> list:
    """Parse the spectrophotometric standards catalog.

    Expects the same fixed-width format as Alex's standards.txt.
    """
    # Regex to handle multi-word names (e.g., "NGC 7293") in fixed-width format.
    # Pattern: name (2+ spaces) RA_h RA_m RA_s (space) Dec_d Dec_m Dec_s (space) Vmag (space) type
    std_re = re.compile(
        r'\s*(.+?)\s{2,}'           # name (greedy until 2+ spaces)
        r'(\d\d)\s+(\d\d)\s+([\d.]+)\s+'  # RA: h m s
        r'([+-]?\d\d)\s+(\d\d)\s+([\d.]+)\s+'  # Dec: d m s
        r'([\d.]+)'                 # V magnitude
        r'(?:\s+(.*))?'             # optional spectral type
    )

    standards = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Star') or line.startswith('-') or line.startswith('h '):
                continue

            m = std_re.match(line)
            if not m:
                continue

            try:
                name = m.group(1).strip()
                ra_str = f"{m.group(2)}:{m.group(3)}:{m.group(4)}"
                dec_str = f"{m.group(5)}:{m.group(6)}:{m.group(7)}"
                vmag = float(m.group(8))
                spec_type = (m.group(9) or '').strip()

                coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
                standards.append({
                    'name': name, 'ra': ra_str, 'dec': dec_str,
                    'coord': coord, 'vmag': vmag, 'spec_type': spec_type,
                })
            except (ValueError, IndexError):
                continue

    return standards


def _best_standard_at(suitable: list, obs_time: Time, config: LLAMASConfig,
                      exclude: set = None) -> Tuple[Optional[dict], Optional[float]]:
    """Pick the highest-scoring suitable standard observable at ``obs_time``.

    Scoring favours low airmass and proximity to the ideal V magnitude.
    Returns (std, airmass) or (None, None) if none is observable.
    """
    exclude = exclude or set()
    best = None
    best_score = -np.inf
    best_am = None
    for std in suitable:
        if std['name'] in exclude:
            continue
        am = _get_airmass(std['coord'], obs_time, config.location)
        if am > config.std_max_airmass or am < 1.0:
            continue
        mag_penalty = abs(std['vmag'] - config.std_ideal_vmag)
        score = -am * 10 - mag_penalty * 2
        if score > best_score:
            best_score = score
            best = std
            best_am = am
    return best, best_am


def _std_dict(std: dict, am: float, time: Optional[Time] = None) -> dict:
    """Build the output dict for a chosen standard."""
    d = {
        'name': std['name'], 'ra': std['ra'], 'dec': std['dec'],
        'vmag': std['vmag'], 'airmass': am,
        'spec_type': std.get('spec_type', ''),
    }
    if time is not None:
        d['time'] = time
    return d


def select_standards(standards_path: str, evening: Time, morning: Time,
                     config: LLAMASConfig = None
                     ) -> Tuple[Optional[dict], Optional[dict], List[dict]]:
    """Select standard stars for start, end, and (long nights) mid-night.

    In addition to the start/end pair, periodic mid-night standards are
    inserted when the night is longer than ``config.standard_interleave_hours``
    (R10) — the design spec calls for 2-3 interleaved standards/night for
    spectrophotometric calibration. The number of mid standards is bounded by
    the night length divided by the cadence, so it stays at the spec's 2-3
    total rather than dozens.

    Parameters
    ----------
    standards_path : str
        Path to standards.txt catalog.
    evening, morning : Time
        Twilight boundaries.
    config : LLAMASConfig, optional

    Returns
    -------
    (start_std, end_std, mid_stds) : tuple
        start_std/end_std are dict or None; mid_stds is a (possibly empty)
        list of dicts. Each dict has: name, ra, dec, vmag, airmass, spec_type
        (mid dicts also carry 'time').
    """
    if config is None:
        config = LLAMAS_CONFIG

    if not Path(standards_path).exists():
        logger.warning("Standards file not found: %s", standards_path)
        return None, None, []

    all_stds = _parse_standards(standards_path)
    suitable = [s for s in all_stds
                if config.std_min_vmag <= s['vmag'] <= config.std_max_vmag]

    # Start standard: ~15 min before evening twilight
    best_start, best_start_am = _best_standard_at(
        suitable, evening - 15 * u.minute, config)

    # End standard: ~15 min after morning twilight (avoid reusing start)
    exclude_end = {best_start['name']} if best_start else set()
    best_end, best_end_am = _best_standard_at(
        suitable, morning + 15 * u.minute, config, exclude=exclude_end)

    start_dict = None
    if best_start:
        start_dict = _std_dict(best_start, best_start_am)
        logger.info("Start standard: %s (V=%.2f, AM=%.2f)",
                     best_start['name'], best_start['vmag'], best_start_am)
    else:
        logger.warning("No suitable start standard found")

    end_dict = None
    if best_end:
        end_dict = _std_dict(best_end, best_end_am)
        logger.info("End standard: %s (V=%.2f, AM=%.2f)",
                     best_end['name'], best_end['vmag'], best_end_am)
    else:
        logger.warning("No suitable end standard found")

    # Mid-night standards on a fixed cadence (R10). Skip entirely if the night
    # is no longer than one cadence interval — the start/end pair suffices.
    mid_dicts = []
    night_hours = (morning - evening).to(u.hour).value
    cadence = config.standard_interleave_hours
    if cadence > 0 and night_hours > cadence:
        # Number of mid points = interior cadence boundaries. e.g. an 8h night
        # at a 3.5h cadence -> boundaries at 3.5h, 7.0h -> 2 mids.
        n_mid = max(1, int(math.floor(night_hours / cadence)) - 1)
        for k in range(1, n_mid + 1):
            obs_time = evening + (k * cadence) * u.hour
            if obs_time >= morning:
                break
            best_mid, best_mid_am = _best_standard_at(
                suitable, obs_time, config)
            if best_mid is None:
                logger.debug("No suitable mid standard at %s", obs_time.iso[11:16])
                continue
            mid_dicts.append(_std_dict(best_mid, best_mid_am, time=obs_time))
            logger.info("Mid standard: %s (V=%.2f, AM=%.2f) @ %s",
                        best_mid['name'], best_mid['vmag'], best_mid_am,
                        obs_time.iso[11:16])

    return start_dict, end_dict, mid_dicts


# ---------------------------------------------------------------------------
# Main scheduling
# ---------------------------------------------------------------------------

def create_schedule(targets: List[Target], evening: Time, morning: Time,
                    moon_phase: str = 'grey',
                    standards_path: str = None,
                    config: LLAMASConfig = None,
                    prioritizer_scores: dict = None,
                    accountant=None,
                    ledger=None,
                    primary_program: Optional[str] = None) -> ObsPlan:
    """Create the complete observing plan for a night.

    Parameters
    ----------
    targets : list of Target
        Observable targets with windows and exposures populated.
    evening, morning : Time
        Twilight boundaries.
    moon_phase : str
        'dark', 'grey', or 'bright'.
    standards_path : str, optional
        Path to standards catalog. Defaults to ref/LDSS_ObsPlan_Generator/standards.txt.
    config : LLAMASConfig, optional
    prioritizer_scores : dict, optional
        {target.name: composite_score} from prioritizer.rank_targets().
        If provided, replaces the default priority-based scoring.
    accountant : TimeAccountant, optional
        If provided, charges scheduled time to each target's program.
    ledger : TargetLedger, optional
        If provided, charges the SAME science time (charged_minutes) to each
        target's per-target integration ledger (W11), so cumulative integration
        accrues across nights.
    primary_program : str, optional
        Tonight's designated PRIMARY program (from the nights CSV). Only the
        primary's must-see (mandatory) targets are guaranteed scheduling via the
        reservation pass; a mandatory target belonging to a NON-primary program
        is demoted to a normal target (still competes in the greedy fill). When
        None (no nights file supplied) ALL mandatory targets are honored —
        backward-compatible behavior.

    Returns
    -------
    ObsPlan
    """
    if config is None:
        config = LLAMAS_CONFIG
    if standards_path is None:
        standards_path = str(DEFAULT_STANDARDS_PATH)

    # Night-primary policy: a mandatory target is HONORED (guaranteed via the
    # reservation pass) only when there is no designated primary tonight
    # (backward-compatible) or its program matches the primary. Non-primary
    # must-see targets are DEMOTED — they go through the moon filter + greedy
    # fill like any normal target.
    def _honored(t):
        if not getattr(t, 'mandatory', False):
            return False
        return primary_program is None or t.program == primary_program

    # Warn once for each demoted (ignored) must-see target.
    for t in targets:
        if getattr(t, 'mandatory', False) and not _honored(t):
            logger.warning(
                "must-see ignored for %s: program %s is not tonight's primary "
                "(%s) — demoted to normal prioritization",
                t.name, t.program, primary_program)

    # 1. Filter targets by moon phase compatibility. HONORED must-see
    # (PI-override) targets BYPASS this filter — the override is non-negotiable
    # on policy — but NOT the physics (a target with no observable window
    # tonight still cannot be scheduled; that is handled in the reservation pass
    # below). Demoted (non-primary) must-see targets fall through to the normal
    # moon-filter path here.
    eligible = []
    for t in targets:
        if _honored(t):
            continue  # honored must-see targets are reserved separately (step 2)
        mc = t.moon_constraint.lower()
        if moon_phase == 'bright' and mc not in ('bright', 'any'):
            continue
        if moon_phase == 'grey' and mc == 'dark':
            continue
        # dark night: all targets eligible
        eligible.append(t)
    logger.info("Moon filter (%s): %d / %d eligible (non-mandatory)",
                moon_phase, len(eligible), len(targets))

    # Sort by transit time for deterministic scheduling
    eligible.sort(key=lambda t: t.transit_time.mjd if t.transit_time else 0)

    scheduled_entries = []
    scheduled_names = set()
    # Reserved (start, end) wall-clock blocks for mandatory targets; the greedy
    # loop must not place anything overlapping these.
    reserved_blocks = []
    unschedulable_mandatory = []

    # Standards are selected UP FRONT so mid-night standards get real reserved
    # blocks the greedy fill must work around — previously they were floating
    # anchors and science was scheduled over them. Start/end standards sit in
    # twilight (before evening / after morning) and consume no dark time.
    std_start, std_end, std_mid = select_standards(
        standards_path, evening, morning, config)
    for smid in (std_mid or []):
        t_mid = smid.get('time')
        if t_mid is None:
            continue
        half = config.std_block_minutes / 2.0 * u.minute
        blk_start = max(evening, t_mid - half)
        blk_end = min(morning, t_mid + half)
        if blk_end > blk_start:
            reserved_blocks.append((blk_start, blk_end))
            smid['start'], smid['end'] = blk_start, blk_end

    def _make_entry(t, start, end):
        """Build a ScheduledEntry for ``t`` over [start, end] (used by the
        mandatory reservation pass). Mirrors the greedy loop's exposure split
        and charging fields so reserved blocks are billed identically."""
        dur = end - start
        total_exp_min = dur.to(u.minute).value - config.overhead_minutes
        total_exp_sec = max(0, int(total_exp_min * 60))
        n_exp = max(1, math.ceil(total_exp_sec / config.max_single_exposure_sec))
        exp_sec = int(round(total_exp_sec / n_exp / config.exposure_round_sec)
                      * config.exposure_round_sec)
        if exp_sec < 10:
            exp_sec = 10
            n_exp = 1
        mid = start + dur / 2
        am = _get_airmass(t.coord, mid, config.location)
        charged_min = exp_sec * n_exp / 60.0 + config.overhead_minutes
        wall_min = dur.to(u.minute).value
        return ScheduledEntry(
            target=t, start=start, end=end, airmass=am,
            exp_str=f"{n_exp}x{exp_sec}s", n_exp=n_exp, exp_sec=exp_sec,
            program=t.program, charged_minutes=charged_min,
            padding_minutes=max(0.0, wall_min - charged_min),
        )

    # 2. Mandatory reservation pass (PI override). Before the greedy fill, place
    # each mandatory target at its best (transit / lowest-airmass) slot within
    # its observable window, in transit-time order, so a non-negotiable target
    # is guaranteed time even if its composite score is low. A mandatory target
    # with NO observable window tonight (never reaches the airmass limit) cannot
    # be scheduled — it is recorded in unschedulable_mandatory and warned about
    # rather than silently dropped. Conflicting mandatory targets are placed in
    # transit order; one that can no longer fit is likewise recorded.
    # Only HONORED must-see targets are reserved here (gated by the night
    # primary). Demoted ones already joined ``eligible`` above and compete in
    # the greedy fill.
    mandatory_targets = [t for t in targets if _honored(t)]
    mandatory_targets.sort(
        key=lambda t: t.transit_time.mjd if t.transit_time else float('inf'))
    for t in mandatory_targets:
        # Physics gate: must have an observable window tonight.
        if t.window_start is None or t.window_end is None:
            logger.warning(
                "MANDATORY target %s is NOT observable tonight (never reaches "
                "the airmass limit) — cannot schedule", t.name)
            unschedulable_mandatory.append(t)
            continue

        exp_min = t.exposure_minutes
        if not math.isfinite(exp_min):
            exp_min = config.fallback_exposure_minutes
        dur = (exp_min + config.overhead_minutes) * u.minute

        # Preferred placement: centered on transit (lowest airmass), clamped to
        # the observable window and the night.
        win_start = max(t.window_start, evening)
        win_end = min(t.window_end, morning)
        # Short-window guarantee: when the observable window is shorter than a
        # full exposure+overhead block, cap the reserved block to the available
        # window rather than dropping the target. An honored must-see with ANY
        # nonzero window is reserved for whatever time it has (the partial
        # integration accrues on the ledger across nights); only a target with
        # no window at all (handled above) stays unschedulable.
        avail = (win_end - win_start).to(u.minute)
        if avail > 0 * u.minute and dur > avail:
            dur = avail
        transit = t.transit_time if t.transit_time is not None else win_start
        start = transit - dur / 2
        if start < win_start:
            start = win_start
        if start + dur > win_end:
            start = win_end - dur

        # Slide forward past any already-reserved block it would overlap.
        placed = False
        for _ in range(len(reserved_blocks) + 1):
            conflict = None
            for (bs, be) in reserved_blocks:
                if start < be and start + dur > bs:
                    conflict = be
                    break
            if conflict is None:
                placed = True
                break
            start = conflict  # slide to just after the conflicting block

        if not placed or start < win_start or start + dur > win_end:
            logger.warning(
                "MANDATORY target %s cannot fit its observable window tonight "
                "(conflicts with another reserved block) — not scheduled",
                t.name)
            unschedulable_mandatory.append(t)
            continue

        end = start + dur
        entry = _make_entry(t, start, end)
        scheduled_entries.append(entry)
        scheduled_names.add(t.name)
        reserved_blocks.append((start, end))
        logger.info("Reserved MANDATORY target %s at %s-%s",
                    t.name, start.iso[11:16], end.iso[11:16])

    # 3. Greedy scheduling loop (fills the remaining time with non-mandatory
    # targets, skipping any reserved mandatory blocks).
    current = evening
    # Coord of the most-recently placed science target; None before the first
    # pick (so the first target incurs no slew penalty).
    prev_coord = None
    # Night-balance bookkeeping: tonight's fair share per program from its
    # fraction of the total allocation in this moon phase.
    fair_share = {}
    sched_science_min = {}
    if accountant is not None and len(getattr(accountant, 'allocations', {})) > 1:
        alloc_tot = 0.0
        for pname, alloc in accountant.allocations.items():
            h = float(alloc.allocated_hours.get(moon_phase, 0.0) or 0.0)
            fair_share[pname] = h
            alloc_tot += h
        if alloc_tot > 0:
            fair_share = {p: h / alloc_tot for p, h in fair_share.items()}
        else:
            fair_share = {}

    def _next_reserved_after(t0):
        """Start time of the earliest reserved block beginning at/after ``t0``,
        or None. Used to skip the greedy cursor over mandatory reservations."""
        starts = [bs for (bs, be) in reserved_blocks if bs >= t0]
        return min(starts, key=lambda s: s.mjd) if starts else None

    def _overlaps_reserved(start, end):
        """True if [start, end] overlaps any reserved mandatory block."""
        for (bs, be) in reserved_blocks:
            if start < be and end > bs:
                return True
        return False

    while current < morning:
        # If the cursor sits inside a reserved mandatory block, jump to its end.
        bumped = True
        while bumped:
            bumped = False
            for (bs, be) in reserved_blocks:
                if bs <= current < be:
                    current = be
                    bumped = True
                    break
        if current >= morning:
            break

        best = None
        best_score = -np.inf
        best_ops_min = 0.0
        best_over = None
        best_over_score = -np.inf
        best_over_ops = 0.0

        for t in eligible:
            if t.name in scheduled_names:
                continue

            exp_min = t.exposure_minutes
            if not math.isfinite(exp_min):
                exp_min = config.fallback_exposure_minutes

            # Per-target operations time: acquisition buffer + slew from the
            # previous pointing (wide-sky plans jump across the hemisphere;
            # the clustered-DDF negligible-slew assumption no longer holds).
            slew_min = 0.0
            sep_deg = 0.0
            if prev_coord is not None:
                sep_deg = prev_coord.separation(t.coord).deg
                slew_min = sep_deg / config.slew_rate_deg_per_min
            ops_min = config.acquisition_buffer_minutes + slew_min
            dur = (exp_min + config.overhead_minutes + ops_min) * u.minute

            # Check target is observable now
            if t.window_start is None or t.window_end is None:
                continue
            if current < t.window_start or current + dur > t.window_end:
                continue
            if current + dur > morning:
                continue
            # Must not collide with a reserved mandatory block.
            if _overlaps_reserved(current, current + dur):
                continue

            # Check airmass at midpoint against the target's effective bounds
            # (per-target ranges are hard constraints; see _am_ok)
            mid = current + dur / 2
            am = _get_airmass(t.coord, mid, config.location)
            if not _am_ok(t, am, config):
                continue

            # Slew score penalty (tie-breaker; the slew TIME is charged in
            # ops_min above).
            slew_pen = sep_deg * SLEW_PENALTY

            # Value density: gently prefer targets that deliver their science
            # in less time (bounded; cannot leapfrog a priority tier).
            density = EXPOSURE_DENSITY_BONUS * min(
                EXPOSURE_DENSITY_REF_MIN / max(exp_min, 1.0), 3.0)

            # Night-balance nudge — PROSPECTIVE and DURATION-AWARE
            # (2026-07-14 redesign after the slot-trace): evaluate the share
            # as it WOULD BE after this pick, so a long pick at a balanced
            # moment feels resistance proportional to the imbalance it is
            # about to create (the retrospective form was exactly zero at
            # that moment, whatever the bonus). Still a bounded soft nudge.
            balance = 0.0
            over_band = False
            if fair_share:
                charged_est = exp_min + config.overhead_minutes
                tot_after = sum(sched_science_min.values()) + charged_est
                share_after = (sched_science_min.get(t.program, 0.0)
                               + charged_est) / tot_after
                deficit_after = fair_share.get(t.program, 0.0) - share_after
                size = min(charged_est / EXPOSURE_DENSITY_REF_MIN, 3.0)
                # imbalance-increasing picks feel size-scaled resistance;
                # underdog picks get the plain boost (not weakened by being
                # small — small picks barely move the split anyway)
                scale = size if deficit_after < 0 else 1.0
                balance = (NIGHT_BALANCE_BONUS
                           * max(-1.0, min(1.0, deficit_after)) * scale)
                # Tolerance band: mark candidates whose pick would push their
                # program beyond fair share + tolerance. They are skipped
                # this slot IF an under-served program has a feasible
                # candidate here (checked after the loop) — never otherwise.
                tol = getattr(config, 'fairness_tolerance', 0.0) or 0.0
                if tol > 0 and deficit_after < -tol:
                    over_band = True

            # Score: use prioritizer if available, else priority-based
            if prioritizer_scores and t.name in prioritizer_scores:
                score = (prioritizer_scores[t.name] - am * 10 - slew_pen
                         + density + balance)
            else:
                score = ((5 - t.priority) * 100 - am * 10 - slew_pen
                         + density + balance)
            if over_band:
                # feasible but over the fairness band: only eligible if no
                # within-band candidate exists this slot
                if score > best_over_score:
                    best_over_score = score
                    best_over = t
                    best_over_ops = ops_min
            elif score > best_score:
                best_score = score
                best = t
                best_ops_min = ops_min

        if best is None and best_over is not None:
            # No within-band candidate is feasible at this slot: the band
            # yields rather than waste sky (feasibility-conditioned).
            best, best_score, best_ops_min = (best_over, best_over_score,
                                              best_over_ops)

        if best is not None:
            exp_min = best.exposure_minutes
            if not math.isfinite(exp_min):
                exp_min = config.fallback_exposure_minutes
            dur = (exp_min + config.overhead_minutes + best_ops_min) * u.minute

            # Gap fill: check if next target's window_start leaves a small gap
            next_ws = None
            for t in eligible:
                if t.name in scheduled_names or t.name == best.name:
                    continue
                if t.window_start is not None and t.window_start > current + dur:
                    if next_ws is None or t.window_start < next_ws:
                        next_ws = t.window_start

            proposed_end = current + dur
            if next_ws is not None and next_ws > proposed_end:
                gap_min = (next_ws - proposed_end).to(u.minute).value
                if 0 < gap_min <= config.gap_fill_max_minutes:
                    extended_end = next_ws
                    # Do not stretch across a reserved mandatory block.
                    next_res = _next_reserved_after(proposed_end)
                    if next_res is not None and next_res < extended_end:
                        extended_end = next_res
                    if (extended_end > proposed_end
                            and extended_end <= best.window_end
                            and extended_end <= morning
                            and not _overlaps_reserved(current, extended_end)):
                        mid_ext = current + (extended_end - current) / 2
                        am_ext = _get_airmass(best.coord, mid_ext, config.location)
                        if _am_ok(best, am_ext, config):
                            dur = extended_end - current

            # Calculate exposure parameters
            # Split total into sub-exposures of at most max_single_exposure_sec
            # each (CR mitigation), per-frame rounded to exposure_round_sec.
            total_exp_min = (dur.to(u.minute).value
                             - config.overhead_minutes - best_ops_min)
            total_exp_sec = int(total_exp_min * 60)
            max_single_sec = config.max_single_exposure_sec
            n_exp = max(1, math.ceil(total_exp_sec / max_single_sec))
            exp_sec = int(round(total_exp_sec / n_exp / config.exposure_round_sec)
                          * config.exposure_round_sec)
            if exp_sec < 10:
                exp_sec = 10
                n_exp = 1

            mid = current + dur / 2
            am = _get_airmass(best.coord, mid, config.location)

            # Science time billed to the program: actual integration
            # (exp_sec x n_exp) + overhead. The wall-clock window (dur) may be
            # larger because of gap-fill stretch above — that slack is padding,
            # not science, so we record it separately and never bill it.
            charged_min = exp_sec * n_exp / 60.0 + config.overhead_minutes
            wall_min = dur.to(u.minute).value
            padding_min = max(0.0, wall_min - charged_min)

            entry = ScheduledEntry(
                target=best,
                start=current,
                end=current + dur,
                airmass=am,
                exp_str=f"{n_exp}x{exp_sec}s",
                n_exp=n_exp,
                exp_sec=exp_sec,
                program=best.program,
                charged_minutes=charged_min,
                padding_minutes=max(0.0, padding_min - best_ops_min),
                ops_minutes=best_ops_min,
            )
            scheduled_entries.append(entry)
            sched_science_min[best.program] = (
                sched_science_min.get(best.program, 0.0) + charged_min)
            scheduled_names.add(best.name)
            prev_coord = best.coord  # anchor slew penalty to this pointing
            current = current + dur
        else:
            # Jump to the next available window — either an eligible target's
            # window start or the end of a reserved mandatory block (so the
            # cursor advances past reservations rather than stalling).
            candidates = [t.window_start for t in eligible
                          if t.name not in scheduled_names
                          and t.window_start is not None
                          and t.window_start > current]
            candidates += [be for (bs, be) in reserved_blocks if be > current]
            if candidates:
                current = min(candidates, key=lambda s: s.mjd)
            else:
                break

    # 3. Post-process: extend the last observation to fill end-of-night gap
    if scheduled_entries:
        entry = scheduled_entries[-1]
        t = entry.target
        end_gap_min = (morning - entry.end).to(u.minute).value
        if end_gap_min > config.gap_fill_max_minutes:
            new_end = min(
                morning,
                entry.end + min(end_gap_min, 5.0) * u.minute,
            )
            if t.window_end is not None:
                new_end = min(new_end, t.window_end)
            # Do not stretch across a reserved mandatory block.
            next_res = _next_reserved_after(entry.end)
            if next_res is not None and next_res < new_end:
                new_end = next_res
            added = (new_end - entry.end).to(u.minute).value
            if added > 0 and not _overlaps_reserved(entry.start, new_end):
                new_mid = entry.start + (new_end - entry.start) / 2
                new_am = _get_airmass(t.coord, new_mid, config.location)
                if _am_ok(t, new_am, config):
                    entry.end = new_end
                    entry.airmass = new_am
                    total_exp_min = (entry.end - entry.start).to(u.minute).value - config.overhead_minutes
                    entry.exp_sec = int(
                        round(total_exp_min * 60 / entry.n_exp
                              / config.exposure_round_sec)
                        * config.exposure_round_sec)
                    if entry.exp_sec < 10:
                        entry.exp_sec = 10
                    entry.exp_str = f"{entry.n_exp}x{entry.exp_sec}s"
                    # The end-of-night stretch is dead-time avoidance: keep
                    # charged_minutes at the science value and fold the added
                    # wall-clock time into padding so the program isn't billed
                    # for it.
                    wall_min = (entry.end - entry.start).to(u.minute).value
                    if math.isfinite(entry.charged_minutes):
                        entry.padding_minutes = max(
                            0.0, wall_min - entry.charged_minutes)

    # Sort by start time
    scheduled_entries.sort(key=lambda e: e.start.mjd)

    # 4. Collect unscheduled as backup
    backup = [t for t in eligible if t.name not in scheduled_names]

    # 5. Charge time to the program accountant and/or the per-target ledger.
    # Both bill the SAME science value (charged_minutes = exposure + overhead),
    # NOT the padded wall-clock window. Padding (gap-fill / end-of-night
    # stretch) is dead-time avoidance and must not be billed.
    if accountant is not None or ledger is not None:
        date_str = str(evening.iso[:10])
        for entry in scheduled_entries:
            if math.isfinite(entry.charged_minutes):
                charged_min = entry.charged_minutes
            else:
                charged_min = (entry.end - entry.start).to(u.minute).value

            if accountant is not None:
                accountant.charge(entry.program, charged_min / 60.0,
                                  moon_phase, date=date_str)

            if ledger is not None:
                t = entry.target
                req_full = getattr(t, 'required_minutes_full', float('nan'))
                required_seconds = (req_full * 60.0
                                    if math.isfinite(req_full) else float('nan'))
                # Charge tonight's phase bucket and record the program so
                # peak/rising integration is tracked separately (W12).
                bucket = phase_bucket(t.delta_t, config.phase_bucket_window_days)
                ledger.charge(t, science_seconds=charged_min * 60.0,
                              date=date_str, mag=t.mag, redshift=t.redshift,
                              required_seconds=required_seconds,
                              phase=bucket, program=t.program)

    # 5b. Shared-ops proration (PI policy, 2026-07-13): standards blocks and
    # per-target ops (slew + acquisition) are pooled and charged to programs
    # PROPORTIONALLY to their science share of the night — a program using
    # two-thirds of the night pays two-thirds of the shared overheads. Logged
    # as its own charge so the audit trail separates science from ops.
    if accountant is not None and scheduled_entries:
        science_by_prog = {}
        ops_pool_min = 0.0
        for entry in scheduled_entries:
            cm = (entry.charged_minutes
                  if math.isfinite(entry.charged_minutes) else 0.0)
            science_by_prog[entry.program] = (
                science_by_prog.get(entry.program, 0.0) + cm)
            ops_pool_min += getattr(entry, 'ops_minutes', 0.0) or 0.0
        n_stds = ((1 if std_start else 0) + (1 if std_end else 0)
                  + len(std_mid or []))
        ops_pool_min += n_stds * config.std_block_minutes
        total_science = sum(science_by_prog.values())
        if ops_pool_min > 0 and total_science > 0:
            date_str = str(evening.iso[:10])
            for prog, sci in science_by_prog.items():
                share = sci / total_science
                accountant.charge(prog, ops_pool_min * share / 60.0,
                                  moon_phase,
                                  date=f"{date_str} shared-ops")
            logger.info("Shared ops pool %.1f min (%d standards + slew/acq) "
                        "prorated by science share: %s",
                        ops_pool_min, n_stds,
                        {p: f"{sci / total_science:.0%}"
                         for p, sci in science_by_prog.items()})

    # 6. Standards were selected up front (step before the reservation pass)
    # so their mid-night blocks are reserved; they are billed only through
    # the shared-ops proration above, never to a single program directly.

    # 7. Build and return ObsPlan
    # Record which scoring path ran (R18) and gather per-target breakdowns
    # (R14) from any target carrying one (set by prioritizer.rank_targets).
    scoring_mode = ('prioritizer' if prioritizer_scores
                    else 'fallback/priority-only')
    score_breakdowns = {}
    for t in list(eligible) + list(backup):
        bd = getattr(t, 'score_breakdown', None)
        if bd:
            score_breakdowns[t.name] = bd

    plan = ObsPlan(
        date=str(evening.iso[:10]),
        evening_twilight=evening,
        morning_twilight=morning,
        moon_phase=moon_phase,
        scheduled=scheduled_entries,
        backup=backup,
        standards_start=std_start,
        standards_end=std_end,
        standards_mid=std_mid,
        scoring_mode=scoring_mode,
        score_breakdowns=score_breakdowns,
        unschedulable_mandatory=unschedulable_mandatory,
    )

    logger.info("Schedule: %d targets, %.0f min, %.0f%% efficiency",
                len(scheduled_entries), plan.scheduled_minutes,
                plan.efficiency * 100)
    return plan
