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

logger = logging.getLogger(__name__)

# Default path to spectrophotometric standards catalog
DEFAULT_STANDARDS_PATH = (
    Path(__file__).parent.parent / 'ref' / 'LDSS_ObsPlan_Generator' / 'standards.txt'
)


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


def _find_window(coord: SkyCoord, evening: Time, morning: Time,
                 location, max_airmass: float) -> tuple:
    """Find the observable window for a target.

    Returns
    -------
    (transit_time, min_airmass, window_start, window_end) or
    (None, None, None, None) if target never reaches max_airmass.
    """
    duration_min = (morning - evening).to(u.minute).value
    times = evening + np.linspace(0, duration_min, 500) * u.minute
    am = _get_airmass_grid(coord, times, location)

    obs_mask = am <= max_airmass
    if not np.any(obs_mask):
        return None, None, None, None

    idx = np.where(obs_mask)[0]
    transit_time = times[np.argmin(am)]
    min_am = float(am.min())
    window_start = times[idx[0]]
    window_end = times[idx[-1]]

    return transit_time, min_am, window_start, window_end


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

def compute_observability(targets: List[Target], evening: Time,
                          morning: Time,
                          config: LLAMASConfig = None) -> List[Target]:
    """Populate observability windows and filter unobservable targets.

    Parameters
    ----------
    targets : list of Target
        Input targets with ra_deg, dec_deg, exposure_minutes populated.
    evening, morning : Time
        Twilight boundaries.
    config : LLAMASConfig, optional

    Returns
    -------
    list of Target
        Observable targets with window fields populated.
    """
    if config is None:
        config = LLAMAS_CONFIG

    observable = []
    for t in targets:
        transit, min_am, ws, we = _find_window(
            t.coord, evening, morning, config.location, config.max_airmass
        )
        if transit is None:
            logger.debug("Not observable: %s", t.name)
            continue

        t.transit_time = transit
        t.min_airmass = min_am
        t.window_start = ws
        t.window_end = we
        t.window_hours = (we - ws).to(u.hour).value

        # Check window is long enough for exposure + overhead
        needed = t.exposure_minutes + config.overhead_minutes
        if not math.isfinite(needed):
            needed = config.fallback_exposure_minutes + config.overhead_minutes
        window_min = t.window_hours * 60.0
        if window_min < needed:
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


def select_standards(standards_path: str, evening: Time, morning: Time,
                     config: LLAMASConfig = None) -> Tuple[Optional[dict], Optional[dict]]:
    """Select standard stars for start and end of night.

    Parameters
    ----------
    standards_path : str
        Path to standards.txt catalog.
    evening, morning : Time
        Twilight boundaries.
    config : LLAMASConfig, optional

    Returns
    -------
    (start_std, end_std) : tuple of dict or None
        Each dict has: name, ra, dec, vmag, airmass, coord, spec_type.
    """
    if config is None:
        config = LLAMAS_CONFIG

    if not Path(standards_path).exists():
        logger.warning("Standards file not found: %s", standards_path)
        return None, None

    all_stds = _parse_standards(standards_path)
    suitable = [s for s in all_stds
                if config.std_min_vmag <= s['vmag'] <= config.std_max_vmag]

    def _score(std, obs_time):
        am = _get_airmass(std['coord'], obs_time, config.location)
        if am > config.std_max_airmass or am < 1.0:
            return None, am
        mag_penalty = abs(std['vmag'] - config.std_ideal_vmag)
        score = -am * 10 - mag_penalty * 2
        return score, am

    # Start standard: ~15 min before evening twilight
    start_time = evening - 15 * u.minute
    best_start = None
    best_start_score = -np.inf
    best_start_am = None

    for std in suitable:
        score, am = _score(std, start_time)
        if score is not None and score > best_start_score:
            best_start_score = score
            best_start = std
            best_start_am = am

    # End standard: ~15 min after morning twilight
    end_time = morning + 15 * u.minute
    best_end = None
    best_end_score = -np.inf
    best_end_am = None

    for std in suitable:
        if best_start and std['name'] == best_start['name']:
            continue
        score, am = _score(std, end_time)
        if score is not None and score > best_end_score:
            best_end_score = score
            best_end = std
            best_end_am = am

    start_dict = None
    if best_start:
        start_dict = {
            'name': best_start['name'], 'ra': best_start['ra'],
            'dec': best_start['dec'], 'vmag': best_start['vmag'],
            'airmass': best_start_am, 'spec_type': best_start.get('spec_type', ''),
        }
        logger.info("Start standard: %s (V=%.2f, AM=%.2f)",
                     best_start['name'], best_start['vmag'], best_start_am)
    else:
        logger.warning("No suitable start standard found")

    end_dict = None
    if best_end:
        end_dict = {
            'name': best_end['name'], 'ra': best_end['ra'],
            'dec': best_end['dec'], 'vmag': best_end['vmag'],
            'airmass': best_end_am, 'spec_type': best_end.get('spec_type', ''),
        }
        logger.info("End standard: %s (V=%.2f, AM=%.2f)",
                     best_end['name'], best_end['vmag'], best_end_am)
    else:
        logger.warning("No suitable end standard found")

    return start_dict, end_dict


# ---------------------------------------------------------------------------
# Main scheduling
# ---------------------------------------------------------------------------

def create_schedule(targets: List[Target], evening: Time, morning: Time,
                    moon_phase: str = 'grey',
                    standards_path: str = None,
                    config: LLAMASConfig = None,
                    prioritizer_scores: dict = None,
                    accountant=None) -> ObsPlan:
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

    Returns
    -------
    ObsPlan
    """
    if config is None:
        config = LLAMAS_CONFIG
    if standards_path is None:
        standards_path = str(DEFAULT_STANDARDS_PATH)

    # 1. Filter targets by moon phase compatibility
    eligible = []
    for t in targets:
        mc = t.moon_constraint.lower()
        if moon_phase == 'bright' and mc not in ('bright', 'any'):
            continue
        if moon_phase == 'grey' and mc == 'dark':
            continue
        # dark night: all targets eligible
        eligible.append(t)
    logger.info("Moon filter (%s): %d / %d eligible",
                moon_phase, len(eligible), len(targets))

    # Sort by transit time for deterministic scheduling
    eligible.sort(key=lambda t: t.transit_time.mjd if t.transit_time else 0)

    # 2. Greedy scheduling loop
    scheduled_entries = []
    scheduled_names = set()
    current = evening

    while current < morning:
        best = None
        best_score = -np.inf

        for t in eligible:
            if t.name in scheduled_names:
                continue

            exp_min = t.exposure_minutes
            if not math.isfinite(exp_min):
                exp_min = config.fallback_exposure_minutes
            dur = (exp_min + config.overhead_minutes) * u.minute

            # Check target is observable now
            if t.window_start is None or t.window_end is None:
                continue
            if current < t.window_start or current + dur > t.window_end:
                continue
            if current + dur > morning:
                continue

            # Check airmass at midpoint
            mid = current + dur / 2
            am = _get_airmass(t.coord, mid, config.location)
            if am > config.max_airmass:
                continue

            # Score: use prioritizer if available, else priority-based
            if prioritizer_scores and t.name in prioritizer_scores:
                score = prioritizer_scores[t.name] - am * 10
            else:
                score = (5 - t.priority) * 100 - am * 10
            if score > best_score:
                best_score = score
                best = t

        if best is not None:
            exp_min = best.exposure_minutes
            if not math.isfinite(exp_min):
                exp_min = config.fallback_exposure_minutes
            dur = (exp_min + config.overhead_minutes) * u.minute

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
                    if extended_end <= best.window_end and extended_end <= morning:
                        mid_ext = current + (extended_end - current) / 2
                        am_ext = _get_airmass(best.coord, mid_ext, config.location)
                        if am_ext <= config.max_airmass:
                            dur = extended_end - current

            # Calculate exposure parameters
            # Split total into sub-exposures of at most 900s (15 min) each
            total_exp_min = dur.to(u.minute).value - config.overhead_minutes
            total_exp_sec = int(total_exp_min * 60)
            max_single_sec = 900
            n_exp = max(1, math.ceil(total_exp_sec / max_single_sec))
            exp_sec = int(round(total_exp_sec / n_exp / 10) * 10)
            if exp_sec < 10:
                exp_sec = 10
                n_exp = 1

            mid = current + dur / 2
            am = _get_airmass(best.coord, mid, config.location)

            entry = ScheduledEntry(
                target=best,
                start=current,
                end=current + dur,
                airmass=am,
                exp_str=f"{n_exp}x{exp_sec}s",
                n_exp=n_exp,
                exp_sec=exp_sec,
                program=best.program,
            )
            scheduled_entries.append(entry)
            scheduled_names.add(best.name)
            current = current + dur
        else:
            # Jump to next available window
            future = [t for t in eligible
                      if t.name not in scheduled_names
                      and t.window_start is not None
                      and t.window_start > current]
            if future:
                next_t = min(future, key=lambda t: t.window_start.mjd)
                current = next_t.window_start
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
            added = (new_end - entry.end).to(u.minute).value
            if added > 0:
                new_mid = entry.start + (new_end - entry.start) / 2
                new_am = _get_airmass(t.coord, new_mid, config.location)
                if new_am <= config.max_airmass:
                    entry.end = new_end
                    entry.airmass = new_am
                    total_exp_min = (entry.end - entry.start).to(u.minute).value - config.overhead_minutes
                    entry.exp_sec = int(round(total_exp_min * 60 / entry.n_exp / 10) * 10)
                    if entry.exp_sec < 10:
                        entry.exp_sec = 10
                    entry.exp_str = f"{entry.n_exp}x{entry.exp_sec}s"

    # Sort by start time
    scheduled_entries.sort(key=lambda e: e.start.mjd)

    # 4. Collect unscheduled as backup
    backup = [t for t in eligible if t.name not in scheduled_names]

    # 5. Charge time if accountant provided
    if accountant is not None:
        date_str = str(evening.iso[:10])
        for entry in scheduled_entries:
            hours = (entry.end - entry.start).to(u.hour).value
            accountant.charge(entry.program, hours, moon_phase, date=date_str)

    # 6. Select standard stars
    std_start, std_end = select_standards(standards_path, evening, morning, config)

    # 7. Build and return ObsPlan
    plan = ObsPlan(
        date=str(evening.iso[:10]),
        evening_twilight=evening,
        morning_twilight=morning,
        moon_phase=moon_phase,
        scheduled=scheduled_entries,
        backup=backup,
        standards_start=std_start,
        standards_end=std_end,
    )

    logger.info("Schedule: %d targets, %.0f min, %.0f%% efficiency",
                len(scheduled_entries), plan.scheduled_minutes,
                plan.efficiency * 100)
    return plan
