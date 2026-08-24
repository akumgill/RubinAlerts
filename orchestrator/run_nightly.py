"""End-to-end nightly orchestration: candidates → schedule → outputs.

Ties together the alert pipeline output (candidates.csv) with time
accounting, composite prioritization, and LLAMAS plan generation.
"""

import json
import logging
import math
from pathlib import Path

from astropy.time import Time

from .accounting import TimeAccountant
from .config import LLAMAS_CONFIG, LLAMASConfig
from .models import ObsPlan
from .normalize import (load_from_rubinalerts, load_targets_csv,
                        estimate_llamas_exposure, load_primary_program)
from .output import write_timeline, write_catalog, write_summary
from .planner import calculate_twilight, compute_observability, create_schedule
from .prioritizer import rank_targets
from .target_ledger import TargetLedger, phase_bucket

logger = logging.getLogger(__name__)


def detect_multi_group_alerts(targets, ledger, accountant=None,
                              config: LLAMASConfig = None) -> list:
    """Flag objects wanted by more than one MAGNETS program.

    An object is "multi-group" if more than one DISTINCT program is associated
    with it, either because tonight's targets at the same sky position belong to
    different programs OR because tonight's program differs from a program that
    already charged the matching ledger entry on a prior night.

    Programs are grouped by ledger entry (coordinate match), so a target renamed
    between nights is still recognised as the same object.

    Returns a list of alert dicts, one per multi-group object:
        {name, ra_deg, dec_deg, programs (sorted), phase_preferences
         {program: pref}, observed_phase (tonight's bucket), same_phase}
    ``same_phase`` is True when every involved program shares one phase
    preference (so no scheduling conflict), False when they differ.
    """
    if config is None:
        config = LLAMAS_CONFIG

    # Group tonight's targets by the SAME OBJECT. A target maps to an existing
    # ledger entry by coordinate match; targets with no ledger entry yet are
    # grouped with each other by coordinate proximity (within the ledger's match
    # radius) so two same-position targets entered tonight cluster even before
    # either is charged. For each group collect the programs seen tonight + the
    # programs already on the matched ledger entry.
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    groups = {}  # key -> {'targets': [...], 'programs': set(), 'coord': SkyCoord}
    for t in targets:
        ent = ledger._find_entry(t)
        t_coord = SkyCoord(ra=t.ra_deg * u.deg, dec=t.dec_deg * u.deg)
        key = None
        if ent is not None:
            key = ent.coord_key
        else:
            # No ledger entry: match against an existing fresh group by sep.
            for gk, g in groups.items():
                if g['coord'] is not None and t_coord.separation(
                        g['coord']).arcsec <= ledger.match_radius_arcsec:
                    key = gk
                    break
            if key is None:
                key = ledger._coord_key(t.ra_deg, t.dec_deg)
        g = groups.setdefault(key, {'targets': [], 'programs': set(),
                                    'coord': t_coord})
        g['targets'].append(t)
        if t.program:
            g['programs'].add(t.program)
        # Fold in programs already recorded on the matched ledger entry.
        if ent is not None:
            g['programs'].update(ent.programs)

    alerts = []
    for key, g in groups.items():
        programs = sorted(g['programs'])
        if len(programs) <= 1:
            continue
        rep = g['targets'][0]  # representative target for name/coords
        prefs = {}
        for p in programs:
            prefs[p] = (accountant.get_phase_preference(p)
                        if accountant is not None else 'peak')
        observed_phase = phase_bucket(rep.delta_t,
                                      config.phase_bucket_window_days)
        same_phase = len(set(prefs.values())) == 1
        alerts.append({
            'name': rep.name,
            'ra_deg': rep.ra_deg,
            'dec_deg': rep.dec_deg,
            'programs': programs,
            'phase_preferences': prefs,
            'observed_phase': observed_phase,
            'same_phase': same_phase,
        })
        logger.warning(
            "MULTI-GROUP target %s wanted by %d programs %s (phase prefs %s; "
            "%s) — observed tonight in %s phase",
            rep.name, len(programs), programs, prefs,
            'same phase' if same_phase else 'DIFFERENT phases',
            observed_phase)
    return alerts


def run_nightly(date: str,
                candidates_path: str,
                allocations_path: str,
                moon_phase: str = 'grey',
                output_dir: str = 'output/',
                standards_path: str = None,
                from_rubinalerts: bool = True,
                config: LLAMASConfig = None,
                target_ledger_path: str = None,
                nights_path: str = None) -> ObsPlan:
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
    nights_path : str, optional
        Path to the observing-nights CSV (date, primary_program). When given,
        only tonight's primary program's must-see (override) targets are
        guaranteed scheduling; everyone else's go through normal prioritization.
        When None, all must-see targets are honored (backward-compatible).

    Returns
    -------
    ObsPlan
    """
    if config is None:
        config = LLAMAS_CONFIG

    # Resolve tonight's primary program (gates the must-see guarantee). None
    # when no nights file is supplied or it lists no row for tonight.
    primary_program = None
    if nights_path:
        primary_program = load_primary_program(nights_path, date)
    if primary_program is None:
        logger.info("No primary program set for %s; all must-see targets "
                    "honored", date)
    else:
        logger.info("Tonight's primary program: %s", primary_program)

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
            program_profiles={p: accountant.get_ranking_profile(p)
                              for p in accountant.allocations},
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

    # 2b. Detect MULTI-GROUP targets (W12): objects wanted by >1 program. Run
    # BEFORE scheduling charges the ledger so a "program A already charged this
    # entry, program B wants it tonight" overlap is caught against the prior
    # state (and tonight's own same-position-different-program overlaps).
    multi_group_alerts = detect_multi_group_alerts(
        targets, ledger, accountant=accountant, config=config)

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
                t.redshift, t.mag, moon_phase, delta_t=t.delta_t,
            )
            if t.moon_constraint == 'any':
                t.moon_constraint = moon_c
        t.required_minutes_full = required_full
        required_minutes_by_target[t.name] = required_full

        # Tonight's phase bucket from the signed time-from-peak. The ledger is
        # queried PER BUCKET so "done at peak" doesn't suppress still-needed
        # rising time (and vice versa).
        bucket = phase_bucket(t.delta_t, config.phase_bucket_window_days)

        # Per-target completeness from the ledger, in tonight's bucket.
        t.cumulative_minutes = ledger.cumulative_seconds(t, phase=bucket) / 60.0
        t.completeness_fraction = ledger.completeness_fraction(
            t, required_full, phase=bucket)

        if ledger.is_satisfied(t, required_full, phase=bucket):
            logger.info("Excluding %s: sufficient %s-phase integration "
                        "(%.0f/%.0f min, %.0f%%)", t.name, bucket,
                        t.cumulative_minutes, required_full,
                        t.completeness_fraction * 100)
            completed.append(t)
            continue

        # Schedule only the remaining time in this phase bucket.
        t.exposure_minutes = ledger.remaining_minutes(
            t, required_full, phase=bucket)
        pending.append(t)

    if not pending:
        logger.warning("All %d targets already satisfied; nothing to schedule",
                       len(targets))
        plan = ObsPlan(date=date, moon_phase=moon_phase)
        plan.completed = completed
        plan.multi_group_alerts = multi_group_alerts
        return plan

    # 4. Calculate twilight
    evening, morning = calculate_twilight(date, config=config)

    # 5. Compute observability (only over the pending, not-yet-satisfied set)
    not_observable: list = []
    observable = compute_observability(pending, evening, morning, config=config,
                                       primary_program=primary_program,
                                       dropped=not_observable)
    if not observable:
        logger.error("No targets observable on %s", date)
        plan = ObsPlan(date=date, moon_phase=moon_phase,
                       evening_twilight=evening, morning_twilight=morning)
        plan.completed = completed
        plan.multi_group_alerts = multi_group_alerts
        plan.not_observable = not_observable
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
        primary_program=primary_program,
    )
    plan.completed = completed
    plan.multi_group_alerts = multi_group_alerts
    plan.not_observable = not_observable

    # 8. Write outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = date.replace('-', '')

    # Multi-group alert sidecar (W12): objects wanted by >1 program.
    if multi_group_alerts:
        alert_path = out_dir / 'multi_group_alerts.json'
        try:
            with open(alert_path, 'w') as f:
                json.dump({'date': date, 'alerts': multi_group_alerts},
                          f, indent=2, sort_keys=True)
            logger.info("Wrote %d multi-group alert(s): %s",
                        len(multi_group_alerts), alert_path)
        except (OSError, TypeError) as e:
            logger.warning("Could not write multi-group alerts: %s", e)

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
