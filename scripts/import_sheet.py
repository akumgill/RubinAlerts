#!/usr/bin/env python
"""Import the collaboration's target spreadsheet into the queue API.

The MAGNETS groups already keep their targets in one shared workbook, one
sheet per instrument/run ("Inst MagellanLLAMAS (Sep 6 & 7;", "LDSS3 12th Sept
- full night", ...). Nobody is going to retype nineteen targets into a web
form while that sheet exists, so the adoption path is to read what they
already maintain rather than ask them to maintain something else.

This is also a standing conformance check on the interface: when the sheet
grows a column or changes a convention, the importer's warnings say how the
real interface has drifted from our model.

The sheet's own conventions, handled here:

  * ``Apparent Mag & Band`` is FREE TEXT: "22 in r", "20.5 in g", and — for
    host/TDE-host targets — surface brightnesses like
    "22.1 mag arcsec^-2 in r at Re = 3.6\"". Parsed into (value, band, kind);
    a surface brightness is submitted as mag_kind=surface_brightness so the
    point-source ETC refuses it instead of quietly sizing it.
  * ``Exposure Time`` is "3x1200s" -> n_exposures + exposure_seconds.
  * ``Priority (1-5; if 4 or 5, added to backup plan)`` -> P1..P5. That maps
    onto our tiers directly, and 4/5 already behave as backup here. P0 has no
    counterpart on the sheet and is never inferred.
  * ``Requested By`` is a PERSON, not a program. The program comes from the
    API key; the person is carried through as requested_by.

Usage:
  python scripts/import_sheet.py "MAGNETS Targets.xlsx" \
      --sheet "Inst MagellanLLAMAS (Sep 6 & 7;" \
      --api https://magnets-collab.onrender.com --key $KEY --instrument LLAMAS
  # see exactly what would be sent, without sending it:
  ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import urllib.request

logger = logging.getLogger(__name__)

# Column headers as they appear in the workbook, after newlines in the header
# cell are folded to spaces. Matching is fuzzy (case-insensitive substring) so a
# renamed or re-wrapped header does not silently drop a field.
FIELD_PATTERNS = {
    "target":   ("target",),
    "ra_deg":   ("ra (j2000 deg)", "ra deg"),
    "dec_deg":  ("dec (j2000 deg)", "dec deg"),
    "ra_sex":   ("ra",),
    "dec_sex":  ("dec",),
    "priority": ("priority",),
    "mag":      ("apparent", "mag & band"),
    "exposure": ("exposure time",),
    "by":       ("requested by",),
    "desc":     ("description",),
    "link":     ("link",),
}

# "22 in r" / "20.5 in g" / "22.1 mag arcsec^-2 in r at Re = 3.6"" / "20 mag/arcs^2"
_SB_MARKERS = ("arcsec", "arcs^2", "arcsec^-2", "arcsec⁻²", "/arcs", "mag/arc")


def parse_mag(raw) -> tuple:
    """Free-text magnitude -> (value, band, kind).

    Returns (nan, 'r', 'point') when nothing numeric is present. The band
    defaults to r only when the text does not name one; guessing a band we were
    not told is how a g magnitude ends up in an r-indexed ETC.
    """
    s = str(raw or "").strip()
    if not s or s.lower() == "nan":
        return float("nan"), "r", "point", False
    kind = ("surface_brightness"
            if any(m in s.lower() for m in _SB_MARKERS) else "point")
    m = re.search(r"(-?\d+\.?\d*)", s)
    value = float(m.group(1)) if m else float("nan")
    b = re.search(r"\bin\s+([ugrizyGVBRIco])\b", s)
    band, said_band = (b.group(1), True) if b else ("r", False)
    return value, band, kind, said_band


def parse_exposure(raw) -> tuple:
    """'3x1200s' -> (3, 1200.0). (None, nan) when unparseable."""
    m = re.match(r"\s*(\d+)\s*[xX]\s*(\d+\.?\d*)", str(raw or ""))
    if not m:
        return None, float("nan")
    return int(m.group(1)), float(m.group(2))


def parse_priority(raw) -> str:
    """Sheet's 1-5 -> P1..P5. Anything else -> P3 (middle), with a warning.
    P0 is OURS (a tonight-guarantee) and is never inferred from a sheet."""
    try:
        n = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        logger.warning("unreadable priority %r -> P3", raw)
        return "P3"
    if not 1 <= n <= 5:
        logger.warning("priority %r out of the sheet's 1-5 range -> P3", raw)
        return "P3"
    return f"P{n}"


def _match_columns(columns) -> dict:
    """Map our field names onto the workbook's actual headers."""
    flat = {str(c).replace("\n", " ").strip(): str(c) for c in columns}
    lowered = {k.lower(): v for k, v in flat.items()}
    found = {}
    for field, pats in FIELD_PATTERNS.items():
        for pat in pats:
            hit = next((orig for low, orig in lowered.items() if pat in low), None)
            if hit is not None:
                found[field] = hit
                break
    return found


def rows_to_items(df, instrument: str = "LLAMAS") -> tuple:
    """Sheet rows -> POST /v1/targets items. Returns (items, warnings)."""
    cols = _match_columns(df.columns)
    warnings = []
    for required in ("target", "priority", "exposure"):
        if required not in cols:
            raise ValueError(f"sheet has no {required!r} column; "
                             f"found {sorted(cols)}")
    missing = [f for f in FIELD_PATTERNS if f not in cols]
    if missing:
        warnings.append(f"columns not found (skipped): {', '.join(missing)}")

    items = []
    for _, r in df.iterrows():
        name = str(r[cols["target"]]).strip()
        if not name or name.lower() == "nan":
            continue
        mag, band, kind, said_band = parse_mag(
            r[cols["mag"]] if "mag" in cols else None)
        n_exp, exp_sec = parse_exposure(r[cols["exposure"]])
        item = {
            "name": name,
            "priority": parse_priority(r[cols["priority"]]),
            "instrument": instrument,
        }
        for key, col in (("ra", "ra_deg"), ("dec", "dec_deg")):
            if col in cols:
                try:
                    item[key] = float(r[cols[col]])
                except (TypeError, ValueError):
                    pass
        if math.isfinite(mag):
            item["mag"] = mag
            item["band"] = band
            item["mag_kind"] = kind
            if not said_band:
                warnings.append(f"{name}: no band stated, assuming r")
            if kind == "surface_brightness":
                warnings.append(
                    f"{name}: magnitude is a SURFACE BRIGHTNESS — the "
                    "point-source ETC cannot size it; exposure comes from the "
                    "sheet")
        else:
            warnings.append(f"{name}: no readable magnitude")
        if n_exp:
            item["n_exposures"] = n_exp
            item["exposure_seconds"] = exp_sec
        else:
            warnings.append(f"{name}: unreadable exposure "
                            f"{r[cols['exposure']]!r} — submission will be "
                            "rejected without one")
        for key, col in (("requested_by", "by"), ("notes", "desc"),
                         ("link", "link")):
            if col in cols:
                v = str(r[cols[col]] or "").strip()
                if v and v.lower() != "nan":
                    item[key] = v
        items.append(item)
    return items, warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workbook", help="path to the .xlsx")
    ap.add_argument("--sheet", default=None,
                    help="sheet name (default: the first one)")
    ap.add_argument("--instrument", default="LLAMAS",
                    choices=("LLAMAS", "LDSS3", "EITHER"))
    ap.add_argument("--api", help="API base URL (omit with --dry-run)")
    ap.add_argument("--key", help="bearer key — its program owns the targets")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the items and the warnings; POST nothing")
    ap.add_argument("--list-sheets", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import pandas as pd
    xl = pd.ExcelFile(args.workbook)
    if args.list_sheets:
        for s in xl.sheet_names:
            print(s)
        return 0
    sheet = args.sheet or xl.sheet_names[0]
    if sheet not in xl.sheet_names:
        logger.error("no sheet %r; workbook has: %s", sheet, xl.sheet_names)
        return 2

    items, warnings = rows_to_items(xl.parse(sheet), args.instrument)
    for w in warnings:
        logger.warning("%s", w)
    logger.info("%d targets from sheet %r", len(items), sheet)

    if args.dry_run:
        print(json.dumps(items, indent=2))
        return 0
    if not (args.api and args.key):
        logger.error("--api and --key are required without --dry-run")
        return 2

    req = urllib.request.Request(
        args.api.rstrip("/") + "/v1/targets",
        data=json.dumps(items).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {args.key}"},
        method="POST")
    with urllib.request.urlopen(req) as resp:
        results = json.loads(resp.read())
    ok = sum(1 for r in results if r.get("status") == "ok")
    for item, r in zip(items, results):
        if r.get("status") != "ok":
            logger.warning("rejected %s: %s", item["name"], r.get("error"))
    logger.info("submitted %d/%d", ok, len(items))
    print(json.dumps({"ok": ok, "total": len(items)}))
    return 0 if ok == len(items) else 1


if __name__ == "__main__":
    raise SystemExit(main())
