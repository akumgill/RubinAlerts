"""Command-line interface for the LLAMAS observing plan generator.

Supports three subcommands:
    plan         — Generate observing plan from target CSV (original behavior)
    run-nightly  — Full nightly run with time accounting and prioritization
    reconcile    — Post-night time reconciliation
"""

import argparse
import logging
import math
import sys
from pathlib import Path

from astropy.time import Time

from .config import LLAMAS_CONFIG
from .normalize import (
    load_targets_csv, load_from_rubinalerts, estimate_llamas_exposure,
)
from .planner import calculate_twilight, compute_observability, create_schedule
from .output import write_timeline, write_catalog, write_summary

logger = logging.getLogger(__name__)


def _default_state_path(output_dir: str) -> str:
    """Canonical location of the time-accounting state file.

    Single convention shared by ``run-nightly`` and ``reconcile``: the state
    JSON always lives at ``<output_dir>/time_accounting.json``. (W11 places its
    per-target ledger beside it using this same helper.)
    """
    return str(Path(output_dir) / 'time_accounting.json')


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )


def _print_plan(plan, date, moon_phase, out_dir):
    """Print plan summary to stdout."""
    print(f"\n{'=' * 60}")
    print(f"LLAMAS Observing Plan: {date} ({moon_phase} moon)")
    print(f"{'=' * 60}")
    print(f"Twilight: {plan.evening_twilight.iso[11:16]} - "
          f"{plan.morning_twilight.iso[11:16]} UT "
          f"({plan.night_duration_hours:.1f} hrs)")
    print(f"Scheduled: {len(plan.scheduled)} targets, "
          f"{plan.scheduled_minutes:.0f} min, "
          f"{plan.efficiency * 100:.0f}% efficiency")
    if plan.backup:
        print(f"Backup: {len(plan.backup)} targets")
    print()

    for i, entry in enumerate(plan.scheduled, 1):
        start_str = entry.start.datetime.strftime('%H:%M')
        end_str = entry.end.datetime.strftime('%H:%M')
        print(f"  {i:2d}. {start_str}-{end_str} {entry.target.name:<16} "
              f"P{entry.target.priority} {entry.exp_str:<10} AM={entry.airmass:.2f}")

    print(f"\nOutputs in: {Path(out_dir).resolve()}")


def cmd_plan(args):
    """Original plan command: generate observing plan from target CSV."""
    _setup_logging(args.verbose)

    # 1. Load targets
    if args.from_rubinalerts:
        targets = load_from_rubinalerts(args.targets)
    else:
        try:
            night_mjd = Time(args.date).mjd
        except Exception:
            night_mjd = float('nan')
        targets = load_targets_csv(args.targets, night_mjd=night_mjd)

    if not targets:
        logger.error("No targets loaded. Exiting.")
        sys.exit(1)

    # 2. Estimate exposures for targets missing them
    for t in targets:
        if not math.isfinite(t.exposure_minutes):
            exp_min, moon_c = estimate_llamas_exposure(
                t.redshift, t.mag, args.moon
            )
            t.exposure_minutes = exp_min
            if t.moon_constraint == 'any':
                t.moon_constraint = moon_c

    # 3. Calculate twilight
    evening, morning = calculate_twilight(args.date)

    # 4. Compute observability
    observable = compute_observability(targets, evening, morning)
    if not observable:
        logger.error("No targets observable. Exiting.")
        sys.exit(1)

    # 5. Create schedule
    plan = create_schedule(
        observable, evening, morning,
        moon_phase=args.moon,
        standards_path=args.standards,
    )

    # 6. Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = args.date.replace('-', '')
    write_timeline(plan, str(out_dir / f'LLAMAS_{date_tag}_timeline.txt'))
    write_catalog(plan, str(out_dir / f'LLAMAS_{date_tag}_catalog.txt'))
    write_summary(plan, str(out_dir / f'LLAMAS_{date_tag}_summary.txt'))

    _print_plan(plan, args.date, args.moon, args.output_dir)


def cmd_run_nightly(args):
    """Full nightly run with time accounting and prioritization."""
    _setup_logging(args.verbose)

    from .run_nightly import run_nightly

    plan = run_nightly(
        date=args.date,
        candidates_path=args.candidates,
        allocations_path=args.allocations,
        moon_phase=args.moon,
        output_dir=args.output_dir,
        standards_path=args.standards,
        from_rubinalerts=not args.csv_format,
    )

    if plan.scheduled:
        _print_plan(plan, args.date, args.moon, args.output_dir)

        # Print budget summary
        from .accounting import TimeAccountant
        state_path = _default_state_path(args.output_dir)
        accountant = TimeAccountant.from_yaml(args.allocations, state_path=state_path)
        print("\nTime Budget:")
        for prog, info in accountant.summary().items():
            used = sum(info['used'].values())
            alloc = sum(info['allocated'].values())
            print(f"  {prog}: {used:.1f}h used / {alloc:.1f}h allocated "
                  f"({info['total_remaining']:.1f}h remaining)")
    else:
        print(f"No targets scheduled for {args.date}")


def cmd_reconcile(args):
    """Post-night time reconciliation."""
    _setup_logging(args.verbose)

    from .accounting import TimeAccountant

    # Resolve the state file. Default to the same location run-nightly wrote it
    # to (<output-dir>/time_accounting.json) so reconcile sees the night's
    # scheduled charges instead of a fresh/empty state (which would make
    # delta = actual - 0 and double-charge the program).
    state_path = args.state if args.state else _default_state_path(args.output_dir)
    logger.info("Reconciling against state file: %s", state_path)

    accountant = TimeAccountant.from_yaml(args.allocations, state_path=state_path)
    delta = accountant.reconcile(
        program=args.program,
        actual_hours=args.actual_hours,
        moon_phase=args.moon,
        date=args.date,
    )

    if abs(delta) < 0.001:
        print(f"No adjustment needed for {args.program} on {args.date}")
    else:
        direction = "added" if delta > 0 else "returned"
        print(f"Reconciled {args.program} on {args.date}: "
              f"{direction} {abs(delta):.2f}h ({args.moon})")

    remaining = accountant.get_remaining(args.program)
    print(f"  {args.program}: {remaining:.1f}h total remaining")


def _add_plan_args(parser):
    """Add common plan arguments."""
    parser.add_argument('--date', required=True, help='Observing date YYYY-MM-DD')
    parser.add_argument('--moon', choices=['dark', 'grey', 'bright'], default='grey',
                        help='Moon phase (default: grey)')
    parser.add_argument('--standards', default=None, help='Standards catalog file')
    parser.add_argument('--output-dir', default='output/', help='Output directory')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable debug logging')


def main():
    parser = argparse.ArgumentParser(
        description='LLAMAS observing plan generator for MAGNETS'
    )
    subparsers = parser.add_subparsers(dest='command')

    # plan subcommand
    plan_parser = subparsers.add_parser('plan', help='Generate plan from target CSV')
    _add_plan_args(plan_parser)
    plan_parser.add_argument('--targets', required=True, help='Target CSV file')
    plan_parser.add_argument('--from-rubinalerts', action='store_true',
                             help='Load from RubinAlerts candidates.csv format')

    # run-nightly subcommand
    nightly_parser = subparsers.add_parser('run-nightly',
                                           help='Full nightly run with time accounting')
    _add_plan_args(nightly_parser)
    nightly_parser.add_argument('--candidates', required=True,
                                help='Candidates CSV (from alert pipeline or manual)')
    nightly_parser.add_argument('--allocations', required=True,
                                help='Allocations YAML file')
    nightly_parser.add_argument('--csv-format', action='store_true',
                                help='Treat input as manual CSV (not RubinAlerts format)')

    # reconcile subcommand
    reconcile_parser = subparsers.add_parser('reconcile',
                                             help='Post-night time reconciliation')
    reconcile_parser.add_argument('--allocations', required=True,
                                  help='Allocations YAML file')
    reconcile_parser.add_argument('--output-dir', default='output/',
                                  help='Output directory holding the state file '
                                       '(default: output/). The state file lives at '
                                       '<output-dir>/time_accounting.json, matching '
                                       'run-nightly.')
    reconcile_parser.add_argument('--state', default=None,
                                  help='Explicit state file path. If omitted, '
                                       'resolves to <output-dir>/time_accounting.json '
                                       '(the same file run-nightly wrote). Always '
                                       'reconcile against the night\'s state, not a '
                                       'fresh file.')
    reconcile_parser.add_argument('--program', required=True,
                                  help='Program name to reconcile')
    reconcile_parser.add_argument('--actual-hours', type=float, required=True,
                                  help='Actual hours observed')
    reconcile_parser.add_argument('--moon', required=True,
                                  choices=['dark', 'grey', 'bright'],
                                  help='Moon phase for the night')
    reconcile_parser.add_argument('--date', required=True,
                                  help='Date to reconcile YYYY-MM-DD')
    reconcile_parser.add_argument('-v', '--verbose', action='store_true')

    # Backward compatibility: if first arg is not a subcommand, assume 'plan'
    known_commands = {'plan', 'run-nightly', 'reconcile'}
    if len(sys.argv) > 1 and sys.argv[1] not in known_commands and sys.argv[1] != '-h' and sys.argv[1] != '--help':
        sys.argv.insert(1, 'plan')

    args = parser.parse_args()

    if args.command == 'plan':
        cmd_plan(args)
    elif args.command == 'run-nightly':
        cmd_run_nightly(args)
    elif args.command == 'reconcile':
        cmd_reconcile(args)
    else:
        parser.print_help()
        sys.exit(1)
