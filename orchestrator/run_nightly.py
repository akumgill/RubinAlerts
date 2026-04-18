"""End-to-end nightly orchestration: candidates → schedule → outputs.

Ties together the alert pipeline output (candidates.csv) with time
accounting, composite prioritization, and LLAMAS plan generation.
"""

import logging
import math
from pathlib import Path

from .accounting import TimeAccountant
from .config import LLAMAS_CONFIG, LLAMASConfig
from .models import ObsPlan
from .normalize import load_from_rubinalerts, load_targets_csv, estimate_llamas_exposure
from .output import write_timeline, write_catalog, write_summary
from .planner import calculate_twilight, compute_observability, create_schedule
from .prioritizer import rank_targets

logger = logging.getLogger(__name__)


def run_nightly(date: str,
                candidates_path: str,
                allocations_path: str,
                moon_phase: str = 'grey',
                output_dir: str = 'output/',
                standards_path: str = None,
                from_rubinalerts: bool = True,
                config: LLAMASConfig = None) -> ObsPlan:
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

    Returns
    -------
    ObsPlan
    """
    if config is None:
        config = LLAMAS_CONFIG

    # 1. Initialize time accounting
    state_path = str(Path(output_dir) / 'time_accounting.json')
    accountant = TimeAccountant.from_yaml(allocations_path, state_path=state_path)
    logger.info("Loaded %d programs from %s", len(accountant.allocations),
                allocations_path)

    # 2. Load targets
    if from_rubinalerts:
        targets = load_from_rubinalerts(
            candidates_path,
            default_program=accountant.default_program,
        )
    else:
        targets = load_targets_csv(candidates_path)
        for t in targets:
            if t.program == 'default':
                t.program = accountant.default_program

    if not targets:
        logger.error("No targets loaded from %s", candidates_path)
        return ObsPlan(date=date, moon_phase=moon_phase)

    logger.info("Loaded %d targets", len(targets))

    # 3. Estimate exposures
    for t in targets:
        if not math.isfinite(t.exposure_minutes):
            exp_min, moon_c = estimate_llamas_exposure(
                t.redshift, t.mag, moon_phase,
            )
            t.exposure_minutes = exp_min
            if t.moon_constraint == 'any':
                t.moon_constraint = moon_c

    # 4. Calculate twilight
    evening, morning = calculate_twilight(date, config=config)

    # 5. Compute observability
    observable = compute_observability(targets, evening, morning, config=config)
    if not observable:
        logger.error("No targets observable on %s", date)
        return ObsPlan(date=date, moon_phase=moon_phase,
                       evening_twilight=evening, morning_twilight=morning)

    # 6. Rank targets with composite scoring
    scores = rank_targets(observable, accountant, evening, morning, config)

    # 7. Create schedule (charges time automatically via accountant)
    plan = create_schedule(
        observable, evening, morning,
        moon_phase=moon_phase,
        standards_path=standards_path,
        config=config,
        prioritizer_scores=scores,
        accountant=accountant,
    )

    # 8. Write outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = date.replace('-', '')

    write_timeline(plan, str(out_dir / f'LLAMAS_{date_tag}_timeline.txt'))
    write_catalog(plan, str(out_dir / f'LLAMAS_{date_tag}_catalog.txt'))
    write_summary(plan, str(out_dir / f'LLAMAS_{date_tag}_summary.txt'),
                  accountant=accountant)

    # 9. Print budget summary
    summary = accountant.summary()
    logger.info("Time budget after scheduling:")
    for prog, info in summary.items():
        logger.info("  %s: %.1fh remaining (factor=%.1f)",
                    prog, info['total_remaining'], info['budget_factor'])

    return plan
