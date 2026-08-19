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
