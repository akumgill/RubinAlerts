"""Export an observing plan into the forms an observer actually uses.

Works on the dashboard dict (``plan`` + ``targets``) that the API already
produces, so no re-run of the scheduler is needed:

- ``catalog_text``   — the instrument catalog the telescope GUI ingests
                       (the deliverable that avoids fat-finger entry). LDSS3
                       and LLAMAS take slightly different "click" formats.
- ``observing_csv``  — a plain observing sheet (seq, UTC, coords, mag, exposure,
                       airmass, note); opens anywhere.
- ``observing_text`` — the LDSS-style timeline sheet for a printable page.

The web layer wires these to a download endpoint + an Export button; nothing
here imports FastAPI or the orchestrator.
"""
from __future__ import annotations

import csv
import io


# ---------------------------------------------------------------------------
# coordinate formatting (no astropy dependency)
# ---------------------------------------------------------------------------
def _ra_hms(deg: float) -> str:
    # decompose from rounded total seconds so ss can never format as 60
    ts = round((deg % 360.0) / 15.0 * 3600.0, 2)
    if ts >= 86400.0:
        ts -= 86400.0
    hh = int(ts // 3600); ts -= hh * 3600
    mm = int(ts // 60); ss = ts - mm * 60
    return f"{hh:02d}:{mm:02d}:{ss:05.2f}"


def _dec_dms(deg: float) -> str:
    sign = "-" if deg < 0 else "+"
    ts = round(abs(deg) * 3600.0, 1)   # arcsec, decompose to avoid ss==60
    dd = int(ts // 3600); ts -= dd * 3600
    mm = int(ts // 60); ss = ts - mm * 60
    return f"{sign}{dd:02d}:{mm:02d}:{ss:04.1f}"


def _notes_by_name(dash: dict) -> dict:
    return {t["name"]: (t.get("notes") or "") for t in dash.get("targets", [])}


# ---------------------------------------------------------------------------
# 1. instrument catalog — what the observing GUI loads
# ---------------------------------------------------------------------------
def catalog_text(dash: dict) -> str:
    """Magellan-style catalog for the scheduled science targets.

    Format per line: ``idx name  RA(hms)  Dec(dms)  2000.0 pmRA pmDec rot``.
    LDSS3 (slit) and LLAMAS (IFU) differ only in the rotator/PA convention;
    both observing GUIs accept this columnar 'click' catalog. Confirm the exact
    rotator field with the instrument scientist before a real night.
    """
    plan = dash.get("plan", {})
    inst = plan.get("instrument", "LLAMAS")
    rot = "HRZ"  # horizon rotator; LDSS3 slit PA is set per-target at the console
    lines = [
        f"# MAGNETS {inst} catalog — {plan.get('date','')}  moon={plan.get('moon','')}",
        f"# {plan.get('n_scheduled',0)} science targets; standards inserted by the "
        f"orchestrator's full plan. Load into the {inst} observing GUI.",
        "# idx  name           RA(2000)      Dec(2000)     epoch pmRA pmDec rot",
    ]
    for i, e in enumerate(plan.get("timeline", []), 1):
        if e.get("ra") is None or e.get("dec") is None:
            continue
        name = str(e["target"]).replace(" ", "_")
        lines.append(
            f"{i:<4} {name:<16} {_ra_hms(e['ra'])}  {_dec_dms(e['dec'])}  "
            f"2000.0 0.0 0.0 {rot}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 2. observing sheet — CSV for the human
# ---------------------------------------------------------------------------
def observing_csv(dash: dict) -> str:
    plan = dash.get("plan", {})
    notes = _notes_by_name(dash)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["seq", "utc", "target", "program", "tier",
                "ra_deg", "dec_deg", "mag", "exp_min", "airmass", "note"])
    for i, e in enumerate(plan.get("timeline", []), 1):
        w.writerow([i, e.get("utc", ""), e.get("target", ""), e.get("program", ""),
                    e.get("tier", ""), e.get("ra", ""), e.get("dec", ""),
                    e.get("mag", ""), e.get("exp_min", ""), e.get("airmass", ""),
                    notes.get(e.get("target", ""), "")])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 3. observing sheet — printable text (LDSS timeline style)
# ---------------------------------------------------------------------------
def observing_text(dash: dict) -> str:
    plan = dash.get("plan", {})
    notes = _notes_by_name(dash)
    out = [
        f"MAGNETS observing plan — {plan.get('date','')}  "
        f"{plan.get('instrument','')}  moon={plan.get('moon','')}  "
        f"dark {plan.get('twilight_start','')}-{plan.get('twilight_end','')} UT",
        f"{plan.get('n_scheduled',0)} targets, "
        f"{plan.get('scheduled_science_hours','?')} h science.  "
        "Nominal sequence — adapt live to conditions (priority order).",
        "",
        f"{'#':<3} {'UTC':<13} {'Target':<14} {'Tier':<4} "
        f"{'RA':<12} {'Dec':<11} {'r':>5} {'Exp':>6} {'X':>5}  Note",
        "-" * 92,
    ]
    for i, e in enumerate(plan.get("timeline", []), 1):
        ra = "" if e.get("ra") is None else _ra_hms(e["ra"])
        dec = "" if e.get("dec") is None else _dec_dms(e["dec"])
        exp = "" if e.get("exp_min") is None else f"{int(e['exp_min'])}m"
        out.append(
            f"{i:<3} {e.get('utc',''):<13} {str(e.get('target','')):<14} "
            f"{e.get('tier',''):<4} {ra:<12} {dec:<11} "
            f"{('' if e.get('mag') is None else e['mag']):>5} {exp:>6} "
            f"{('' if e.get('airmass') is None else e['airmass']):>5}  "
            f"{notes.get(e.get('target',''),'')}"
        )
    return "\n".join(out) + "\n"
