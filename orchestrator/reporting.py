"""Reporting for LLAMAS observing plans and season progress.

Generates time usage reports, season progress summaries, and
enhanced observing plan summaries with budget information.
"""

import logging
from pathlib import Path

from .accounting import TimeAccountant
from .models import ObsPlan

logger = logging.getLogger(__name__)


def write_time_report(accountant: TimeAccountant, plan: ObsPlan,
                      path: str) -> None:
    """Per-program time usage report for a single night.

    Shows hours charged tonight and remaining budget per program.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"Time Report: {plan.date} ({plan.moon_phase} moon)")
    lines.append("=" * 60)
    lines.append("")

    # Tonight's charges by program
    tonight_charges = {}
    for entry in accountant.charge_log:
        if entry.get('date') == plan.date and entry.get('type') == 'schedule':
            prog = entry['program']
            tonight_charges[prog] = tonight_charges.get(prog, 0.0) + entry['hours']

    if tonight_charges:
        lines.append("Tonight's charges:")
        for prog, hours in sorted(tonight_charges.items()):
            lines.append(f"  {prog:<20} {hours:.2f}h ({plan.moon_phase})")
        lines.append(f"  {'TOTAL':<20} {sum(tonight_charges.values()):.2f}h")
        lines.append("")

    # Budget overview
    summary = accountant.summary()
    lines.append("Budget Status:")
    lines.append(f"  {'Program':<20} {'Dark':>6} {'Grey':>6} {'Bright':>6} "
                 f"{'Total':>6} {'Factor':>6}")
    lines.append(f"  {'-' * 56}")

    for prog, info in summary.items():
        r = info['remaining']
        lines.append(
            f"  {prog:<20} {r['dark']:>5.1f}h {r['grey']:>5.1f}h "
            f"{r['bright']:>5.1f}h {info['total_remaining']:>5.1f}h "
            f"{info['budget_factor']:>5.1f}"
        )
    lines.append("")
    lines.append("=" * 60)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info("Wrote time report: %s", path)


def write_season_report(accountant: TimeAccountant, path: str) -> None:
    """Cumulative season progress report from charge log.

    Shows per-program usage over time, burn rate, and projected exhaustion.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"Season Progress: {accountant.semester}")
    lines.append("=" * 60)
    lines.append("")

    summary = accountant.summary()

    # Overall progress
    total_alloc = sum(
        sum(info['allocated'].values()) for info in summary.values()
    )
    total_used = sum(
        sum(info['used'].values()) for info in summary.values()
    )
    total_remain = total_alloc - total_used
    pct = (total_used / total_alloc * 100) if total_alloc > 0 else 0

    lines.append(f"Overall: {total_used:.1f}h / {total_alloc:.1f}h "
                 f"({pct:.0f}% used, {total_remain:.1f}h remaining)")
    lines.append("")

    # Per-program detail
    lines.append("Per-program breakdown:")
    lines.append(f"  {'Program':<20} {'Alloc':>6} {'Used':>6} {'Remain':>6} {'%Used':>6}")
    lines.append(f"  {'-' * 44}")

    for prog, info in summary.items():
        alloc = sum(info['allocated'].values())
        used = sum(info['used'].values())
        remain = info['total_remaining']
        pct_used = (used / alloc * 100) if alloc > 0 else 0
        lines.append(
            f"  {prog:<20} {alloc:>5.1f}h {used:>5.1f}h "
            f"{remain:>5.1f}h {pct_used:>5.0f}%"
        )
    lines.append("")

    # Night-by-night from charge log
    nights = {}
    for entry in accountant.charge_log:
        if entry.get('type') != 'schedule':
            continue
        date = entry.get('date', '?')
        if date not in nights:
            nights[date] = {'total': 0.0, 'programs': {}}
        nights[date]['total'] += entry['hours']
        prog = entry['program']
        nights[date]['programs'][prog] = (
            nights[date]['programs'].get(prog, 0.0) + entry['hours']
        )

    if nights:
        lines.append("Night-by-night:")
        lines.append(f"  {'Date':<12} {'Hours':>6} {'Programs'}")
        lines.append(f"  {'-' * 50}")
        for date in sorted(nights):
            n = nights[date]
            progs = ', '.join(f"{p}={h:.1f}h"
                              for p, h in sorted(n['programs'].items()))
            lines.append(f"  {date:<12} {n['total']:>5.1f}h {progs}")

        # Burn rate
        n_nights = len(nights)
        avg_per_night = total_used / n_nights if n_nights > 0 else 0
        lines.append("")
        lines.append(f"Burn rate: {avg_per_night:.1f}h/night "
                     f"({n_nights} nights observed)")
        if avg_per_night > 0:
            nights_left = total_remain / avg_per_night
            lines.append(f"At this rate: ~{nights_left:.0f} nights remaining")

    lines.append("")
    lines.append("=" * 60)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info("Wrote season report: %s", path)
