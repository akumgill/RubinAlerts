"""End-to-end nightly orchestration: candidates → schedule → outputs.

Ties together the alert pipeline output (candidates.csv) with time
accounting, composite prioritization, and LLAMAS plan generation.
"""

import logging
import math
from pathlib import Path

from astropy.time import Time

from .accounting import TimeAccountant
from .config import LLAMAS_CONFIG, LLAMASConfig
from .models import ObsPlan
from .normalize import load_from_rubinalerts, load_targets_csv, estimate_llamas_exposure
from .output import write_timeline, write_catalog, write_summary
from .planner import calculate_twilight, compute_observability, create_schedule
from .prioritizer import rank_targets
from .target_ledger import TargetLedger

logger = logging.getLogger(__name__)


def run_nightly(date: str,
                candidates_path: str,
                allocations_path: str,
                moon_phase: str = 'grey',
                output_dir: str = 'output/',
                standards_path: str = None,
                from_rubinalerts: bool = True,
                config: LLAMASConfig = None,
                target_ledger_path: str = None) -> ObsPlan:
    """Full nightly run: load candidates, apply budgets, generate plan.

    Parameters
    ----------
    date : str
        Observing date YYYY-MM-DD.
    candidates_path : str
        Path to target list (candidates.csv or manual CSV).
    allocations_path : str
        Path to allocations YAML file.
    moon_phase : str
        'dark', 'grey', or 'bright'.
    output_dir : str
        Directory for output files.
    standards_path : str, optional
        Path to standards catalog.
    from_rubinalerts : bool
        If True, load using RubinAlerts format (default).
    config : LLAMASConfig, optional
    target_ledger_path : str, optional
        Path to the per-target integration ledger JSON (W11). Defaults to
        ``<output_dir>/target_ledger.json``. Tracks cumulative integration per
        target across nights so already-sufficient targets are excluded and
        partially-observed ones are scheduled only for their remaining time.

    Returns
    -------
    ObsPlan
    """
    if config is None:
        config = LLAMAS_CONFIG

    # 1. Initialize time accounting and the per-target integration ledger.
    state_path = str(Path(output_dir) / 'time_accounting.json')
    accountant = TimeAccountant.from_yaml(allocations_path, state_path=state_path)
    logger.info("Loaded %d programs from %s", len(accountant.allocations),
                allocations_path)

    if target_ledger_path is None:
        target_ledger_path = str(Path(output_dir) / 'target_ledger.json')
    ledger = TargetLedger.load(target_ledger_path)

    # 2. Load targets
    if from_rubinalerts:
        targets = load_from_rubinalerts(
            candidates_path,
            default_program=accountant.default_program,
        )
    else:
        try:
            night_mjd = Time(date).mjd
        except Exception:
            night_mjd = float('nan')
        targets = load_targets_csv(
            candidates_path,
            night_mjd=night_mjd,
            default_program=accountant.default_program,
        )

    if not targets:
        logger.error("No targets loaded from %s", candidates_path)
        return ObsPlan(date=date, moon_phase=moon_phase)

    logger.info("Loaded %d targets", len(targets))

    # 3. Estimate the FULL required exposure per target, then consult the
    # per-target ledger. The ledger lets us (a) skip targets that already have
    # sufficient cumulative integration and (b) schedule partially-observed
    # targets only for their REMAINING time. We therefore store the full
    # required exposure on required_minutes_full and set exposure_minutes to the
    # remaining time to observe tonight.
    required_minutes_by_target = {}
    pending = []
    completed = []
    for t in targets:
        # Full required exposure for this target tonight.
        if math.isfinite(t.exposure_minutes):
            required_full = t.exposure_minutes
            moon_c = t.moon_constraint
        else:
            required_full, moon_c = estimate_llamas_exposure(
                t.redshift, t.mag, moon_phase,
            )
            if t.moon_constraint == 'any':
                t.moon_constraint = moon_c
        t.required_minutes_full = required_full
        required_minutes_by_target[t.name] = required_full

        # Per-target completeness from the ledger.
        t.cumulative_minutes = ledger.cumulative_seconds(t) / 60.0
        t.completeness_fraction = ledger.completeness_fraction(t, required_full)

        if ledger.is_satisfied(t, required_full):
            logger.info("Excluding %s: sufficient integration (%.0f/%.0f min, "
                        "%.0f%%)", t.name, t.cumulative_minutes, required_full,
                        t.completeness_fraction * 100)
            completed.append(t)
            continue

        # Schedule only the remaining time.
        t.exposure_minutes = ledger.remaining_minutes(t, required_full)
        pending.append(t)

    if not pending:
        logger.warning("All %d targets already satisfied; nothing to schedule",
                       len(targets))
        plan = ObsPlan(date=date, moon_phase=moon_phase)
        plan.completed = completed
        return plan

    # 4. Calculate twilight
    evening, morning = calculate_twilight(date, config=config)

    # 5. Compute observability (only over the pending, not-yet-satisfied set)
    observable = compute_observability(pending, evening, morning, config=config)
    if not observable:
        logger.error("No targets observable on %s", date)
        plan = ObsPlan(date=date, moon_phase=moon_phase,
                       evening_twilight=evening, morning_twilight=morning)
        plan.completed = completed
        return plan

    # 6. Rank targets with composite scoring. The ledger folds each target's
    # completeness into the science core (breakdowns are attached to each target
    # and flow through create_schedule onto the plan for write_summary).
    scores, _breakdowns = rank_targets(
        observable, accountant, evening, morning, config,
        moon_phase=moon_phase, ledger=ledger,
        required_minutes_by_target=required_minutes_by_target)

    # 7. Create schedule (charges time to the accountant AND the per-target
    # ledger automatically).
    plan = create_schedule(
        observable, evening, morning,
        moon_phase=moon_phase,
        standards_path=standards_path,
        config=config,
        prioritizer_scores=scores,
        accountant=accountant,
        ledger=ledger,
    )
    plan.completed = completed

    # 8. Write outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = date.replace('-', '')

    write_timeline(plan, str(out_dir / f'LLAMAS_{date_tag}_timeline.txt'))
    write_catalog(plan, str(out_dir / f'LLAMAS_{date_tag}_catalog.txt'))
    write_summary(plan, str(out_dir / f'LLAMAS_{date_tag}_summary.txt'),
                  accountant=accountant, ledger=ledger)

    # 9. Print budget summary
    summary = accountant.summary()
    logger.info("Time budget after scheduling:")
    for prog, info in summary.items():
        logger.info("  %s: %.1fh remaining (factor=%.1f)",
                    prog, info['total_remaining'], info['budget_factor'])

    return plan
