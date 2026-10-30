"""Observing-file serializers for operator batches (item G).

Three output layers for a 4-6 target batch ("go make me a file that does
that plan" — Chris). Two have REAL in-repo reference formats and are matched
column-for-column; the third is provisional:

  tcs_catalog   — Magellan TCS catalog, modeled on ref/march_obs_run/
                  catalog.cat (a genuine catalog from a real run):
                  TAB-separated, 15 fields: index, name, RA HH:MM:SS.SS,
                  Dec ±DD:MM:SS.SS, equinox 2000.0, pm 0.0 0.0, rotator
                  offset -62.5, mode HRZ, then two empty guide-star slots
                  (00:00:00.0 +00:00:00 2000.0). NOTE: orchestrator/output.py's
                  older writer mixed tabs and spaces — the real file is
                  all-tab, which is what we emit here.
  plan_sheet    — the observer plan-sheet convention of
                  ref/LDSS_ObsPlan_Generator/example_targets.txt:
                  TAB-separated: name, RA hms, Dec dms, ra deg, dec deg,
                  priority, date, instrument (the example's unused 'N/A'
                  column, repurposed as agreed), mag, '3x900s' exposure
                  triplet, note, link.
  llamas_macro  — LLAMAS instrument macro commands: FORMAT UNKNOWN pending
                  Rob Simcoe / LCO docs. Kept behind this serializer
                  interface with a clearly-marked provisional implementation
                  so swapping in the real dialect touches ONE function.

All serializers take plain row dicts:
  {name, ra, dec (deg), mag, priority ('P1'.. or int), n_exp, exp_sec,
   notes, instrument, link?}
Exposures are ALWAYS triplet-shaped upstream (3 x N s cosmic-ray median
protocol) but any (n_exp, exp_sec) is rendered faithfully.
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

TCS_ROTATOR_OFFSET = "-62.5"
TCS_ROTATOR_MODE = "HRZ"
_TCS_GUIDE_SLOT = ("00:00:00.0", "+00:00:00", "2000.0")


def ra_hms(ra_deg: float) -> str:
    """RA degrees -> HH:MM:SS.SS (catalog.cat convention, 2dp seconds)."""
    h = (ra_deg % 360.0) / 15.0
    hh = int(h)
    m = (h - hh) * 60.0
    mm = int(m)
    ss = (m - mm) * 60.0
    if ss >= 59.995:                      # carry, avoid '60.00'
        ss = 0.0
        mm += 1
        if mm == 60:
            mm = 0
            hh = (hh + 1) % 24
    return f"{hh:02d}:{mm:02d}:{ss:05.2f}"


def dec_dms(dec_deg: float) -> str:
    """Dec degrees -> ±DD:MM:SS.SS (catalog.cat convention, 2dp seconds)."""
    sign = "-" if dec_deg < 0 else "+"
    d = abs(dec_deg)
    dd = int(d)
    m = (d - dd) * 60.0
    mm = int(m)
    ss = (m - mm) * 60.0
    if ss >= 59.995:
        ss = 0.0
        mm += 1
        if mm == 60:
            mm = 0
            dd += 1
    return f"{sign}{dd:02d}:{mm:02d}:{ss:05.2f}"


def exp_str(row: dict) -> str:
    """'3x600s' from (n_exp, exp_sec)."""
    return f"{int(row.get('n_exp') or 1)}x{int(round(row.get('exp_sec') or 0))}s"


def _pri_num(priority) -> str:
    """'P2' -> '2'; ints pass through (the plan sheet uses bare numbers)."""
    s = str(priority or "3")
    return s[1:] if s.upper().startswith("P") and s[1:].isdigit() else s


def tcs_catalog(rows: list[dict]) -> str:
    """Magellan TCS catalog — column-for-column ref/march_obs_run/catalog.cat."""
    lines = []
    for i, r in enumerate(rows, 1):
        fields = (str(i), str(r["name"]), ra_hms(r["ra"]), dec_dms(r["dec"]),
                  "2000.0", "0.0", "0.0", TCS_ROTATOR_OFFSET, TCS_ROTATOR_MODE,
                  *_TCS_GUIDE_SLOT, *_TCS_GUIDE_SLOT)
        lines.append("\t".join(fields))
    return "\n".join(lines) + "\n"


def plan_sheet(rows: list[dict], date: str, instrument: str = "LLAMAS") -> str:
    """Observer plan sheet — the LDSS_ObsPlan_Generator column convention."""
    lines = []
    for r in rows:
        mag = r.get("mag")
        lines.append("\t".join((
            str(r["name"]),
            ra_hms(r["ra"]), dec_dms(r["dec"]),
            f"{r['ra']:.6f}", f"{r['dec']:.6f}",
            _pri_num(r.get("priority")),
            str(date),
            str(r.get("instrument") or instrument),
            "N/A" if mag is None else f"{mag:.1f}",
            exp_str(r),
            str(r.get("notes") or "").replace("\t", " ") or "—",
            str(r.get("link") or "N/A"),
        )))
    return "\n".join(lines) + "\n"


def llamas_macro(rows: list[dict]) -> str:
    """LLAMAS instrument macro — PROVISIONAL serializer.

    The real macro dialect is unknown (pending Rob Simcoe / LCO docs); this
    emits a commented command block per target (pointing reference + the CR
    triplet) so an operator can read it, and so swapping in the real command
    grammar means rewriting ONLY this function.
    """
    lines = ["# LLAMAS macro — FORMAT PROVISIONAL pending Simcoe/LCO docs.",
             "# One block per target: point, acquire, expose the CR triplet.",
             ""]
    for i, r in enumerate(rows, 1):
        n = int(r.get("n_exp") or 1)
        sec = int(round(r.get("exp_sec") or 0))
        lines += [
            f"# --- target {i}: {r['name']} ---",
            f"# pointing: {ra_hms(r['ra'])} {dec_dms(r['dec'])} (J2000)",
            f"# LLAMAS: {n} x {sec}s — FORMAT PROVISIONAL pending Simcoe/LCO docs",
            "",
        ]
    return "\n".join(lines)


def night_plan(rows: list[dict]) -> str:
    """Observer night plan in the collaboration's own convention.

    Matches the table Yize Dong circulates for LLAMAS nights (2026-09-06/07):

        |    name |          ra |          dec |  mag | exposure | nexp | exptime |                   start |

    Deliberately NOT the ``plan_sheet`` format. That one follows the 12-column
    LDSS_ObsPlan_Generator convention and feeds a tool; this one is what a human
    observer reads at the telescope, so it carries an ABSOLUTE start time and
    drops everything they do not act on (decimal coords, priority, links).

    Each row needs name/ra/dec/mag/n_exp/exp_sec plus ``start`` — a UTC
    ``datetime`` (or an ISO string). Rows with no start sort last and show '—'.
    """
    hdr = ("name", "ra", "dec", "mag", "exposure", "nexp", "exptime", "start")
    body = []
    for r in rows:
        start = r.get("start")
        if hasattr(start, "strftime"):
            start = start.strftime("%Y-%m-%d %H:%M:%S")
        mag = r.get("mag")
        n_exp = int(r.get("n_exp") or 1)
        exp_sec = float(r.get("exp_sec") or 0)
        body.append((
            str(r.get("name") or ""),
            ra_hms(r["ra"]), dec_dms(r["dec"]),
            "—" if mag is None else f"{float(mag):g}",
            f"{n_exp}x{int(round(exp_sec))}",
            f"{n_exp}.0", f"{exp_sec:.1f}",
            str(start) if start else "—",
        ))
    w = [max(len(hdr[i]), *(len(b[i]) for b in body)) if body else len(hdr[i])
         for i in range(len(hdr))]
    line = lambda cells: "| " + " | ".join(
        c.rjust(w[i]) for i, c in enumerate(cells)) + " |"
    return "\n".join([line(hdr)] + [line(b) for b in body]) + "\n"


# --- Magellan CfA observing request package (LaTeX) -------------------------
# The observer (Yuri Beletsky) asks for the plan as the CfA LaTeX template,
# compiled to PDF. The package is a FOLDER, not a single file — the reference
# Yize circulated for 2026-10-03 is:
#
#     Dong_Magellan_1003_2026/
#         LDSS_20261003_Dong.pdf          <- the compiled template
#         ObsPlan/Dong_Magellan_1003_catalog
#         ObsPlan/Dong_Magellan_1003_timeline
#         Finders/                        <- one finder chart per target
#
# so the PDF's "Observing Catalog" and "Finder Charts" sections are POINTERS to
# sibling files rather than embedded content. We already generate the catalog
# (tcs_catalog) and the timeline (night_plan); this renders the document that
# wraps them.

# Longest acceptable single sub-exposure. The collaboration's own plans use
# 900 s and 1200 s and vary the COUNT, never the length: Yize's Sep 6/7 and
# Oct 3 packages run 2x900, 3x900, 3x1200, 4x1200, 5x1200. A 2140 s sub-
# exposure (which naive total/3 produces for a mag-22 target) would be
# unusable for cosmic rays and tracking.
MAX_SUB_EXPOSURE_SEC = 1200
MIN_SUB_EXPOSURES = 3          # the CR-rejection triplet is the floor


def split_subexposures(total_minutes, max_sub_sec: int = MAX_SUB_EXPOSURE_SEC,
                       min_n: int = MIN_SUB_EXPOSURES) -> tuple:
    """Total integration -> (n_exposures, seconds each), rounded to 10 s.

    Splits by COUNT, keeping each sub-exposure at or under ``max_sub_sec``,
    with at least ``min_n`` so cosmic rays can always be median-filtered.
    """
    try:
        total_sec = float(total_minutes) * 60.0
    except (TypeError, ValueError):
        return min_n, 0.0
    if not (total_sec > 0):
        return min_n, 0.0
    n = max(min_n, math.ceil(total_sec / float(max_sub_sec)))
    return n, max(10.0, round(total_sec / n / 10.0) * 10.0)


_ORDINALS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}


def _tex_escape(s) -> str:
    """Escape the characters that bite in target names and notes.

    Underscores are the live hazard: the collaboration writes photometry
    entries as ``2026adpo_image``, which is a subscript in LaTeX and a compile
    error outside maths mode.
    """
    out = []
    for ch in str(s):
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        elif ch == "\\":
            out.append("\\textbackslash{}")
        else:
            out.append(ch)
    return "".join(out)


def _priority_phrase(priority) -> str:
    """'P2' / 2 -> '(2nd Priority)'; blank for an unknown tier."""
    n = _pri_num(priority)
    try:
        return f"({_ORDINALS[int(n)]} Priority)"
    except (ValueError, KeyError):
        return ""


def _target_block(r: dict) -> list:
    """One target as the reference package writes it: a bullet carrying name,
    priority and anticipated magnitude, then the numbered observation(s)."""
    mag = r.get("mag")
    mag_s = "" if mag is None else f" mag $\\sim$ {float(mag):g}"
    if r.get("is_standard"):
        head = (f"Spectroscopic Standard Star {_tex_escape(r['name'])}"
                + (f" (V mag = {float(mag):g})" if mag is not None else ""))
    else:
        head = f"{_tex_escape(r['name'])} {_priority_phrase(r.get('priority'))}{mag_s}"
    lines = [f"\\item {head}", "\\begin{enumerate}"]
    for obs in (r.get("observations") or [f"Spectrum: {exp_str(r)}"]):
        lines.append(f"\\item {_tex_escape(obs)}")
    lines += ["\\end{enumerate}", ""]
    return lines


def request_package(nights: list, meta: dict) -> str:
    """Render the Magellan CfA observing request package as LaTeX.

    ``nights`` is one entry per observing night::

        {"date": "2026-11-02",            # ISO; shown as "November 02, 2026"
         "label": "...",                  # optional subsection heading override
         "targets": [ {name, priority, mag, n_exp, exp_sec,
                       is_standard?, observations?: ["Spectrum: 3x900s", ...]} ]}

    Content from ``meta`` is treated as AUTHORED LaTeX and passed through
    verbatim: it is prose the PI wrote, and it legitimately contains maths like
    ``$\\sim 20{,}000$``. Only GENERATED content — target names out of the
    queue, filenames, observation strings — is escaped, because that is where a
    stray underscore (``2026adpo_image``) comes from.

    ``meta`` carries the standing content the template asks for — see
    ``ref/request_package_stubbs.yaml`` for our program's copy. Everything in
    it is prose the PI owns, deliberately NOT generated: an observing request
    is a document a person signs their name to.

    Required meta keys: pi, instrument, contact_name, contact_email,
    contact_phone, science_program, catalog_file. Optional: instruments_text,
    strategy, strategy_subsections ({title: body}), finders_dir, backup_intro,
    backup (same target shape as a night), calibrations_day, calibrations_night,
    timeline_intro, data_transfer.

    Returns the .tex source. Compile with pdflatex (twice is unnecessary — the
    document has no cross-references or table of contents).
    """
    need = ("pi", "instrument", "contact_name", "contact_email",
            "contact_phone", "science_program", "catalog_file")
    missing = [k for k in need if not meta.get(k)]
    if missing:
        raise ValueError(f"request_package: meta is missing {missing}")
    if not nights:
        raise ValueError("request_package: no nights given")

    def _pretty(d):
        from datetime import date as _d
        try:
            return _d.fromisoformat(str(d)).strftime("%B %d, %Y")
        except (TypeError, ValueError):
            return str(d)

    dates = [_pretty(n.get("date")) for n in nights]
    title_dates = dates[0] if len(dates) == 1 else f"{dates[0]} -- {dates[-1]}"
    inst = _tex_escape(meta["instrument"])

    L = [
        "\\documentclass[a4paper]{article}",
        "\\usepackage[colorlinks=true, allcolors=blue]{hyperref}",
        "\\usepackage[a4paper,top=2cm,bottom=2cm,left=2cm,right=2cm,"
        "marginparwidth=1.75cm]{geometry}",
        f"\\title{{{title_dates} {inst} {_tex_escape(meta['pi'])} "
        "Observing Request Package}",
        "\\date{}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\\section{Requested Instruments}",
        (meta.get("instruments_text")
         or f"{meta['instrument']} will be used for all targets."),
        "",
        "\\section{Contact Info}",
        "",
        f"{meta['contact_name']} \\\\",
        f"{meta['contact_email']} \\\\",
        f"{meta['contact_phone']}",
        "",
        "\\section{Description of Science Program}",
        meta["science_program"],
        "",
        "\\section{Observing Strategy and Expected Products}",
        meta.get("strategy", ""),
        "",
    ]
    for title, body in (meta.get("strategy_subsections") or {}).items():
        L += [f"\\subsection{{{title}}}", body, ""]

    L += [
        "\\section{Observing Catalog}",
        f"See file called \\texttt{{{_tex_escape(meta['catalog_file'])}}}.",
        "",
        "\\section{Finder Charts}",
        f"See folder called \\texttt{{{_tex_escape(meta.get('finders_dir', 'Finders'))}}}.",
        "",
        "\\section{Backup Program}",
    ]
    backup = meta.get("backup") or []
    if meta.get("backup_intro"):
        L += [meta["backup_intro"], ""]
    if backup:
        L.append("\\begin{itemize}")
        for r in backup:
            L += _target_block(r)
        L.append("\\end{itemize}")
    else:
        L.append("No backup targets are queued for this run.")
    L += ["", "\\section{Calibration Plan}", ""]
    for sub, items in (("Daytime Calibrations", meta.get("calibrations_day")),
                       ("Nighttime Calibrations", meta.get("calibrations_night"))):
        if not items:
            continue
        L += [f"\\subsection{{{sub}}}", "\\begin{itemize}"]
        L += [f"\\item {i}" for i in items]
        L += ["\\end{itemize}", ""]

    L += ["\\section{Observing Timeline}",
          (meta.get(
              "timeline_intro",
              "Listed in desired observing order. Please modify as necessary "
              "due to weather.")),
          ""]
    for night, pretty in zip(nights, dates):
        L += [f"\\subsection{{{_tex_escape(night.get('label') or pretty)}}}",
              "\\begin{itemize}"]
        for r in night.get("targets", []):
            L += _target_block(r)
        L += ["\\end{itemize}", ""]

    L += ["\\section{Data Transfer}",
          (meta.get(
              "data_transfer",
              "Please upload the resulting data to Google Drive and send us a "
              "link. Live transfer is not required; uploading in the morning "
              "is fine.")),
          "",
          "\\end{document}",
          ""]
    return "\n".join(L)
