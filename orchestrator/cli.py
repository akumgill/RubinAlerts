"""Command-line interface for the LLAMAS observing plan generator."""

import argparse
import logging
import math
import sys
from pathlib import Path

from .config import LLAMAS_CONFIG
from .normalize import (
    load_targets_csv, load_from_rubinalerts, estimate_llamas_exposure,
)
from .planner import calculate_twilight, compute_observability, create_schedule
from .output import write_timeline, write_catalog, write_summary

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='LLAMAS observing plan generator for MAGNETS'
    )
    parser.add_argument('--date', required=True, help='Observing date YYYY-MM-DD')
    parser.add_argument('--targets', required=True, help='Target CSV file')
    parser.add_argument('--moon', choices=['dark', 'grey', 'bright'], default='grey',
                        help='Moon phase (default: grey)')
    parser.add_argument('--standards', default=None, help='Standards catalog file')
    parser.add_argument('--output-dir', default='output/', help='Output directory')
    parser.add_argument('--from-rubinalerts', action='store_true',
                        help='Load targets from RubinAlerts candidates.csv format')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    # 1. Load targets
    if args.from_rubinalerts:
        targets = load_from_rubinalerts(args.targets)
    else:
        targets = load_targets_csv(args.targets)

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

    # 7. Print summary to stdout
    print(f"\n{'=' * 60}")
    print(f"LLAMAS Observing Plan: {args.date} ({args.moon} moon)")
    print(f"{'=' * 60}")
    print(f"Twilight: {evening.iso[11:16]} - {morning.iso[11:16]} UT "
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

    print(f"\nOutputs in: {out_dir.resolve()}")
