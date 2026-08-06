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


def _windows_by_name(dash: dict) -> dict:
    """name -> observability-window fields (obs_start/end/best, min_airmass,
    window_note) so exports can show when a target is up, not just a slot."""
    return {t["name"]: t for t in dash.get("targets", [])}


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
    win = _windows_by_name(dash)
    buf = io.StringIO()
    w = csv.writer(buf)
    # nominal_utc is a suggested pace only; observable_start/end + best are the
    # real freedom the observer works within (reorder as conditions dictate).
    w.writerow(["seq", "nominal_utc", "target", "program", "tier",
                "observable_start", "observable_end", "best_utc", "min_airmass",
                "when", "ra_deg", "dec_deg", "mag", "exp_min", "note"])
    for i, e in enumerate(plan.get("timeline", []), 1):
        t = win.get(e.get("target", ""), {})
        w.writerow([i, e.get("utc", ""), e.get("target", ""), e.get("program", ""),
                    e.get("tier", ""), t.get("obs_start", ""), t.get("obs_end", ""),
                    t.get("obs_best", ""), t.get("min_airmass", ""),
                    t.get("window_note", ""), e.get("ra", ""), e.get("dec", ""),
                    e.get("mag", ""), e.get("exp_min", ""),
                    notes.get(e.get("target", ""), "")])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 3. observing sheet — printable text (LDSS timeline style)
# ---------------------------------------------------------------------------
def observing_text(dash: dict) -> str:
    plan = dash.get("plan", {})
    notes = _notes_by_name(dash)
    win = _windows_by_name(dash)
    out = [
        f"MAGNETS observing plan — {plan.get('date','')}  "
        f"{plan.get('instrument','')}  moon={plan.get('moon','')}  "
        f"dark {plan.get('twilight_start','')}-{plan.get('twilight_end','')} UT",
        f"{plan.get('n_scheduled',0)} targets, "
        f"{plan.get('scheduled_science_hours','?')} h science.  "
        "Priority order + observable window — reorder freely; do the 'sets' "
        "targets first. 'Nominal' UTC is a suggested pace only.",
        "",
        f"{'#':<3} {'Target':<14} {'Tier':<4} {'Observable':<15} {'Best(X)':<14} "
        f"{'r':>5} {'Exp':>6}  When / note",
        "-" * 100,
    ]
    for i, e in enumerate(plan.get("timeline", []), 1):
        t = win.get(e.get("target", ""), {})
        obs = (f"{t.get('obs_start')}-{t.get('obs_end')}"
               if t.get("obs_start") and t.get("obs_end") else "—")
        best = (f"{t.get('obs_best')} X{t.get('min_airmass')}"
                if t.get("obs_best") else "—")
        exp = "" if e.get("exp_min") is None else f"{int(e['exp_min'])}m"
        note = t.get("window_note", "")
        extra = notes.get(e.get("target", ""), "")
        if extra:
            note = f"{note} · {extra}" if note else extra
        out.append(
            f"{i:<3} {str(e.get('target','')):<14} {e.get('tier',''):<4} "
            f"{obs:<15} {best:<14} "
            f"{('' if e.get('mag') is None else e['mag']):>5} {exp:>6}  "
            f"nominal ~{e.get('utc','')}  {note}"
        )
    return "\n".join(out) + "\n"
