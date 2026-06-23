"""Output writers for LLAMAS observing plans.

Generates timeline, Magellan TCS catalog, and human-readable summary files.
"""

import json
import logging
import math
from pathlib import Path

import astropy.units as u

from .models import ObsPlan, Target

logger = logging.getLogger(__name__)


def write_timeline(plan: ObsPlan, path: str) -> None:
    """Write the observing timeline in the standard format.

    Format matches Alex's LDSS timeline:
    #  Target     UT          Observation     Comments
    """
    lines = []
    lines.append("#  Target     UT          Observation     Comments")

    catalog_idx = {}
    idx = 1

    # Standard at start
    if plan.standards_start:
        s = plan.standards_start
        catalog_idx[s['name']] = idx
        ev_str = plan.evening_twilight.datetime.strftime('%H:%M') if plan.evening_twilight else '??:??'
        lines.append(
            f"{idx} {s['name']} before {ev_str} "
            f"spec: 2x30s V={s['vmag']:.2f} standard"
        )
        idx += 1

    # Science targets, with mid-night standards interleaved by time so the
    # calibration cadence is visible in the timeline.
    mids = sorted(plan.standards_mid or [],
                  key=lambda s: s['time'].mjd if s.get('time') is not None else 0)
    mid_i = 0

    def _emit_mid_before(t):
        """Emit any mid-night standard whose target time precedes ``t``."""
        nonlocal mid_i, idx
        while mid_i < len(mids) and mids[mid_i].get('time') is not None and \
                t is not None and mids[mid_i]['time'].mjd <= t.mjd:
            s = mids[mid_i]
            catalog_idx[s['name']] = idx
            ts = s['time'].datetime.strftime('%H:%M')
            lines.append(
                f"{idx} {s['name']} near {ts} "
                f"spec: 2x30s V={s['vmag']:.2f} standard"
            )
            idx += 1
            mid_i += 1

    for entry in plan.scheduled:
        _emit_mid_before(entry.start)
        catalog_idx[entry.target.name] = idx
        start_str = entry.start.datetime.strftime('%H:%M') if entry.start else '??:??'
        end_str = entry.end.datetime.strftime('%H:%M') if entry.end else '??:??'
        comment = f"P{entry.target.priority}"
        if entry.target.notes:
            comment += f" {entry.target.notes}"
        lines.append(
            f"{idx} {entry.target.name} {start_str} - {end_str} "
            f"spec: {entry.exp_str} {comment}"
        )
        idx += 1

    # Any remaining mid standards (times after the last science target)
    _emit_mid_before(plan.morning_twilight)

    # Standard at end
    if plan.standards_end:
        s = plan.standards_end
        catalog_idx[s['name']] = idx
        last_end = plan.scheduled[-1].end if plan.scheduled else plan.morning_twilight
        end_str = last_end.datetime.strftime('%H:%M') if last_end else '??:??'
        lines.append(
            f"{idx} {s['name']} after {end_str} "
            f"spec: 2x30s V={s['vmag']:.2f} standard"
        )
        idx += 1

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info("Wrote timeline: %s", path)


def write_catalog(plan: ObsPlan, path: str) -> None:
    """Write a Magellan TCS-format catalog file.

    Format: ref name RA Dec 2000.0 0.0 0.0 -62.5 HRZ ...
    """
    lines = []
    idx = 1

    def _add_entry(name, ra_hms, dec_dms):
        nonlocal idx
        lines.append(
            f"{idx} {name}\t{ra_hms}\t{dec_dms} 2000.0 0.0 0.0 -62.5 HRZ "
            f"00:00:00.0   +00:00:00   2000.0   00:00:00.0   +00:00:00   2000.0"
        )
        idx += 1

    # Standard at start
    if plan.standards_start:
        s = plan.standards_start
        _add_entry(s['name'], s['ra'], s['dec'])

    # Scheduled targets
    for entry in plan.scheduled:
        t = entry.target
        _add_entry(t.name, t.ra_hms, t.dec_dms)

    # Backup targets
    for t in plan.backup:
        _add_entry(t.name, t.ra_hms, t.dec_dms)

    # Standard at end
    if plan.standards_end:
        s = plan.standards_end
        _add_entry(s['name'], s['ra'], s['dec'])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info("Wrote catalog: %s (%d entries)", path, idx - 1)


def _phase_breakdown_str(ledger, target) -> str:
    """Render a target's per-phase cumulative integration, e.g.
    'peak=44m rising=12m'. Empty string if no ledger or no record."""
    if ledger is None:
        return ''
    ent = ledger._find_entry(target)
    if ent is None or not ent.cumulative_seconds_by_phase:
        return ''
    parts = [f"{ph}={secs / 60.0:.0f}m"
             for ph, secs in sorted(ent.cumulative_seconds_by_phase.items())
             if secs]
    return ' '.join(parts)


def write_summary(plan: ObsPlan, path: str, accountant=None, ledger=None) -> None:
    """Write a human-readable summary of the observing plan.

    When a ``ledger`` (W11 TargetLedger) is supplied, the per-target table gains
    Cumul (cumulative integration minutes) and Done% (completeness) columns, the
    per-phase integration breakdown (W12) is shown for scheduled/completed
    targets, and a "Completed" section lists targets excluded for having
    sufficient integration time (``plan.completed``).

    Multi-group targets (W12: objects wanted by >1 program) are rendered in a
    dedicated alert section.
    """
    lines = []

    lines.append("=" * 70)
    lines.append("LLAMAS Observing Plan Summary")
    lines.append("=" * 70)
    lines.append("")

    # Night info
    lines.append(f"Date:              {plan.date}")
    if plan.evening_twilight:
        lines.append(f"Evening twilight:  {plan.evening_twilight.iso[11:16]} UT")
    if plan.morning_twilight:
        lines.append(f"Morning twilight:  {plan.morning_twilight.iso[11:16]} UT")
    lines.append(f"Night duration:    {plan.night_duration_hours:.1f} hours")
    lines.append(f"Moon phase:        {plan.moon_phase}")
    # R18: record which scoring path produced the ranking so the scale of the
    # numbers below is unambiguous.
    lines.append(f"Scoring mode:      {plan.scoring_mode}")
    lines.append("")
    # R8: P-labels are WITHIN-NIGHT RELATIVE (tonight's quartiles), not absolute
    # science classes. A P1 tonight may be weaker than a P4 on a richer night.
    lines.append("Note: P1-P4 are WITHIN-NIGHT RELATIVE priorities (tonight's "
                 "ranking), not absolute classes.")
    lines.append("")

    # Schedule summary
    lines.append(f"Scheduled targets: {len(plan.scheduled)}")
    lines.append(f"Backup targets:    {len(plan.backup)}")
    lines.append(f"Total time:        {plan.scheduled_minutes:.0f} min "
                 f"({plan.scheduled_minutes / 60:.1f} hrs)")
    lines.append(f"Efficiency:        {plan.efficiency * 100:.1f}%")
    lines.append("")

    # Priority breakdown
    priority_counts = {}
    for entry in plan.scheduled:
        p = entry.target.priority
        priority_counts[p] = priority_counts.get(p, 0) + 1
    if priority_counts:
        lines.append("Priority breakdown:")
        for p in sorted(priority_counts):
            lines.append(f"  P{p}: {priority_counts[p]}")
        lines.append("")

    # Standards
    if plan.standards_start:
        s = plan.standards_start
        lines.append(f"Start standard:    {s['name']} (V={s['vmag']:.2f}, AM={s['airmass']:.2f})")
    for s in (plan.standards_mid or []):
        ts = s['time'].iso[11:16] if s.get('time') is not None else '??:??'
        lines.append(f"Mid standard:      {s['name']} (V={s['vmag']:.2f}, "
                     f"AM={s['airmass']:.2f}) @ {ts} UT")
    if plan.standards_end:
        s = plan.standards_end
        lines.append(f"End standard:      {s['name']} (V={s['vmag']:.2f}, AM={s['airmass']:.2f})")
    lines.append("")

    # Scheduled targets detail. With a ledger, append per-target cumulative
    # integration (Cumul, minutes) and completeness (Done%) columns.
    show_ledger = ledger is not None
    width = 90 if show_ledger else 70
    lines.append("-" * width)
    header = (f"{'#':<4} {'Target':<18} {'UT Start':>8} {'UT End':>8} "
              f"{'Exp':>10} {'AM':>5} {'P':>2}")
    if show_ledger:
        header += f" {'Cumul':>7} {'Done%':>6}"
    lines.append(header)
    lines.append("-" * width)

    for i, entry in enumerate(plan.scheduled, 1):
        start_str = entry.start.datetime.strftime('%H:%M') if entry.start else '??:??'
        end_str = entry.end.datetime.strftime('%H:%M') if entry.end else '??:??'
        row = (
            f"{i:<4} {entry.target.name:<18} {start_str:>8} {end_str:>8} "
            f"{entry.exp_str:>10} {entry.airmass:>5.2f} {entry.target.priority:>2}"
        )
        if show_ledger:
            t = entry.target
            cumul = (f"{t.cumulative_minutes:>6.0f}m"
                     if math.isfinite(t.cumulative_minutes) else f"{'-':>7}")
            done = (f"{t.completeness_fraction * 100:>5.0f}%"
                    if math.isfinite(t.completeness_fraction) else f"{'-':>6}")
            row += f" {cumul} {done}"
        lines.append(row)

    lines.append("-" * width)
    lines.append("")

    # Completed targets excluded for sufficient integration (W11). With a
    # ledger, append the per-phase integration breakdown (W12).
    if plan.completed:
        lines.append("Completed (excluded — sufficient integration):")
        for t in plan.completed:
            cumul = (f"{t.cumulative_minutes:.0f} min"
                     if math.isfinite(t.cumulative_minutes) else "? min")
            done = (f"{t.completeness_fraction * 100:.0f}%"
                    if math.isfinite(t.completeness_fraction) else "?")
            row = (f"  {t.name:<18} P{t.priority} "
                   f"cumulative={cumul} done={done}")
            by_phase = _phase_breakdown_str(ledger, t)
            if by_phase:
                row += f"  [{by_phase}]"
            lines.append(row)
        lines.append("")

    # Per-phase integration breakdown for scheduled targets (W12).
    if ledger is not None and plan.scheduled:
        phase_rows = []
        for entry in plan.scheduled:
            by_phase = _phase_breakdown_str(ledger, entry.target)
            if by_phase:
                phase_rows.append(f"  {entry.target.name:<18} {by_phase}")
        if phase_rows:
            lines.append("Per-phase integration (scheduled):")
            lines.extend(phase_rows)
            lines.append("")

    # Per-target composite-score breakdown (R14): mirror the alert-pipeline
    # merit breakdown so a PI can reconstruct any ranking. Only present when
    # the prioritizer ran (breakdowns attached to the plan).
    if plan.score_breakdowns:
        from .prioritizer import (SCIENCE_SCALE, OBSERVABILITY_BONUS,
                                  KEYWORD_SCALE)
        lines.append("-" * 78)
        lines.append(
            f"Composite score breakdown ({SCIENCE_SCALE:g} x sci x budget x "
            f"phase + {OBSERVABILITY_BONUS:g} x obs + {KEYWORD_SCALE:g} x kw):")
        lines.append(f"{'Target':<18} {'sci':>5} {'budget':>6} {'phase':>5} "
                     f"{'obs':>5} {'kw':>6} {'total':>7}")
        lines.append("-" * 78)
        # Order by total descending for readability.
        ordered = sorted(plan.score_breakdowns.items(),
                         key=lambda kv: kv[1].get('total', 0.0), reverse=True)
        for name, bd in ordered:
            lines.append(
                f"{name:<18} {bd.get('science', 0.0):>5.2f} "
                f"{bd.get('budget', 0.0):>6.2f} {bd.get('phase', 0.0):>5.2f} "
                f"{bd.get('observability', 0.0):>5.2f} "
                f"{bd.get('keyword_adj', 0.0):>6.2f} "
                f"{bd.get('total', 0.0):>7.1f}"
            )
        lines.append("-" * 78)
        lines.append("")

    # Multi-group alerts (W12): objects wanted by more than one program.
    if plan.multi_group_alerts:
        lines.append("=" * 70)
        lines.append("MULTI-GROUP TARGETS (wanted by >1 program)")
        lines.append("=" * 70)
        for a in plan.multi_group_alerts:
            phase_note = ('SAME phase preference' if a.get('same_phase')
                          else 'DIFFERENT phase preferences')
            lines.append(f"  {a['name']:<18} programs={', '.join(a['programs'])}")
            prefs = a.get('phase_preferences', {})
            pref_str = ', '.join(f"{p}->{prefs[p]}" for p in a['programs'])
            lines.append(f"  {'':<18} {pref_str}  ({phase_note})")
            lines.append(f"  {'':<18} observed tonight in {a.get('observed_phase', '?')} phase")
        lines.append("")

    # Backup targets
    if plan.backup:
        lines.append("Backup targets:")
        for t in plan.backup:
            mag_str = f"mag={t.mag:.1f}" if math.isfinite(t.mag) else "mag=?"
            lines.append(f"  {t.name:<18} P{t.priority} {mag_str}")
        lines.append("")

    # Time accounting (if available)
    if accountant is not None:
        summary = accountant.summary()
        if summary:
            lines.append("Time Budget:")
            lines.append(f"  {'Program':<20} {'Used':>6} {'Alloc':>6} {'Remain':>6} {'Factor':>6}")
            for prog, info in summary.items():
                used = sum(info['used'].values())
                alloc = sum(info['allocated'].values())
                lines.append(
                    f"  {prog:<20} {used:>5.1f}h {alloc:>5.1f}h "
                    f"{info['total_remaining']:>5.1f}h {info['budget_factor']:>5.1f}"
                )
            lines.append("")

    lines.append("=" * 70)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info("Wrote summary: %s", path)

    # R14: persist the per-target breakdown as a JSON sidecar in the output dir
    # (mirrors the alert-pipeline merit breakdown). Always written so consumers
    # have a stable artifact; empty when the fallback path ran.
    sidecar = out_path.parent / 'score_breakdown.json'
    payload = {
        'date': plan.date,
        'moon_phase': plan.moon_phase,
        'scoring_mode': plan.scoring_mode,
        'breakdowns': plan.score_breakdowns,
    }
    try:
        with open(sidecar, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        logger.info("Wrote score breakdown: %s", sidecar)
    except (OSError, TypeError) as e:
        logger.warning("Could not write score breakdown sidecar: %s", e)
