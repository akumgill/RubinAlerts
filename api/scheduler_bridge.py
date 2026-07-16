"""Bridge from the API queue to the LLAMAS orchestrator.

The plan preview does NOT re-implement scheduling: it materializes the current
queue into the manual-CSV schema the orchestrator already accepts, runs the
real nightly scheduler as a dry run, and reshapes the resulting ObsPlan into
the preview response (per-program requested hours, a program-tagged timeline,
and the overflow of submitted-but-unscheduled targets).
"""
from __future__ import annotations

import csv
import logging
import math
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# API priority tier -> (orchestrator integer priority, mandatory?). P0 is the
# "observe tonight" guarantee: top ordinary priority PLUS the mandatory
# reservation the orchestrator already implements. P1>P2>P3 are ordinary.
_TIER_MAP = {"P0": (1, True), "P1": (1, False), "P2": (2, False), "P3": (3, False)}


def _moon_phase_for(date: str) -> str:
    """Derive dark/grey/bright from the date's lunar illumination (astropy).
    The submitter never supplies this — it's a property of the night."""
    try:
        from astropy.time import Time
        from astropy.coordinates import get_body, get_sun
        t = Time(date) + 0.5  # local midnight-ish
        elong = get_sun(t).separation(get_body("moon", t)).radian
        illum = (1 - math.cos(elong)) / 2
        return "dark" if illum < 0.25 else "grey" if illum < 0.65 else "bright"
    except Exception as e:
        logger.warning("moon derivation failed (%s); assuming grey", e)
        return "grey"


def _materialize_csv(targets, path: str) -> None:
    cols = ["name", "ra", "dec", "priority", "mag", "redshift",
            "exposure", "program", "keywords", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, t in enumerate(targets):
            pri, mandatory = _TIER_MAP.get(t.priority, (3, False))
            w.writerow({
                "name": t.name or f"{t.program}-{t.id}",
                "ra": f"{t.canonical_ra:.6f}",
                "dec": f"{t.canonical_dec:.6f}",
                "priority": pri,
                "mag": "" if not math.isfinite(t.mag) else f"{t.mag:.3f}",
                "redshift": "" if not math.isfinite(t.redshift) else f"{t.redshift:.4f}",
                "exposure": "" if not math.isfinite(t.exposure_minutes)
                            else f"{t.exposure_minutes:.1f}",
                "program": t.program,
                "keywords": "mandatory" if mandatory else "",
                "notes": t.notes or "",
            })


def _tier_of(target, id_to_tier) -> str:
    return id_to_tier.get((target.program, target.name), "P?")


def preview_plan(service, date: str, moon: str = None) -> dict:
    """Run the orchestrator over the live queue; return the preview dict."""
    from orchestrator.run_nightly import run_nightly

    active = service.active()
    if not active:
        return {"date": date, "moon": moon or _moon_phase_for(date),
                "requested_hours": {}, "timeline": [], "overflow": [],
                "note": "queue is empty"}

    moon = moon or _moon_phase_for(date)

    # requested hours per program (before scheduling)
    summary = service.queue_summary()
    requested = {p: v["requested_hours"] for p, v in summary.items()}

    # name/program -> tier, to tag timeline entries with the submitted tier
    id_to_tier = {}
    for t in active:
        key_name = t.name or f"{t.program}-{t.id}"
        id_to_tier[(t.program, key_name)] = t.priority

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = str(Path(tmp) / "queue.csv")
        _materialize_csv(active, csv_path)
        plan = run_nightly(
            date=date, candidates_path=csv_path,
            allocations_path=service.allocations_path,
            moon_phase=moon, output_dir=tmp, from_rubinalerts=False,
        )

    def _hhmm(t):
        return t.datetime.strftime("%H:%M") if t is not None else "--:--"

    timeline = []
    for e in plan.scheduled:
        tgt = e.target
        timeline.append({
            "utc": f"{_hhmm(e.start)}-{_hhmm(e.end)}",
            "target": tgt.name,
            "program": e.program or tgt.program,
            "tier": _tier_of(tgt, id_to_tier),
            "ra": round(tgt.ra_deg, 4), "dec": round(tgt.dec_deg, 4),
            "mag": None if not math.isfinite(tgt.mag) else round(tgt.mag, 1),
            "exp_min": None if not math.isfinite(e.charged_minutes)
                       else round(e.charged_minutes, 0),
            "airmass": None if not math.isfinite(e.airmass) else round(e.airmass, 2),
        })

    overflow = []
    for tgt in plan.backup:
        overflow.append({
            "target": tgt.name, "program": tgt.program,
            "tier": _tier_of(tgt, id_to_tier),
            "reason": "below the line for tonight (time/priority)",
        })
    for tgt in getattr(plan, "unschedulable_mandatory", []):
        overflow.append({
            "target": tgt.name, "program": tgt.program,
            "tier": _tier_of(tgt, id_to_tier),
            "reason": "P0 guaranteed but not observable tonight (never reaches airmass limit)",
        })

    sci_hours = sum(e.charged_minutes for e in plan.scheduled
                    if math.isfinite(e.charged_minutes)) / 60.0
    return {
        "date": date, "moon": moon,
        "requested_hours": requested,
        "scheduled_science_hours": round(sci_hours, 2),
        "n_scheduled": len(plan.scheduled),
        "timeline": timeline,
        "overflow": overflow,
    }
