#!/usr/bin/env python
"""Enqueue spectrophotometric standards as airmass-binned pseudo-targets.

Standards are calibration measurements AT a specific airmass, so each standard
is submitted once per airmass bin as a pseudo-target named
``<name>@am<lo>-<hi>`` (e.g. ``GD71@am1.0-1.3``) carrying that bin as its
[airmass_min, airmass_max] hard scheduling range (stamped #5). The queue's
coordinate dedupe keeps bins distinct (dedup is per airmass-range spec), and
re-running the script upserts rather than duplicates.

Input: a CSV (or JSON list) of standards with columns/keys
  name, ra, dec, mag[, exposure_minutes]
(ra/dec in decimal degrees). See ref/standards_example.csv.

Usage:
  python scripts/enqueue_standards.py ref/standards_example.csv \
      --api http://localhost:8000 --key <bearer> --exposure-minutes 6
  # custom bins / tier:
  ... --bins 1.0-1.3,1.3-1.7,1.7-2.3 --priority P2 --dry-run

The program charged is whichever the --key resolves to. Exposure is required
by the queue API: give --exposure-minutes or a per-row exposure_minutes
column. The actual standards list arrives later — this script is the
ingestion vehicle.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_BINS = "1.0-1.3,1.3-1.7,1.7-2.3"


def parse_bins(spec: str) -> list[tuple[float, float]]:
    """'1.0-1.3,1.3-1.7' -> [(1.0, 1.3), (1.3, 1.7)]; validated."""
    bins = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo_s, hi_s = part.split("-")
        lo, hi = float(lo_s), float(hi_s)
        if not (1.0 <= lo < hi):
            raise ValueError(f"bad airmass bin {part!r}: need 1.0 <= lo < hi")
        bins.append((lo, hi))
    if not bins:
        raise ValueError("no airmass bins given")
    return bins


def load_standards(path: str) -> list[dict]:
    """CSV (name, ra, dec, mag[, exposure_minutes]) or a JSON list of dicts."""
    if path.lower().endswith(".json"):
        with open(path) as f:
            rows = json.load(f)
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            "name": str(r["name"]).strip(),
            "ra": float(r["ra"]), "dec": float(r["dec"]),
            "mag": float(r["mag"]) if r.get("mag") not in (None, "") else None,
            "exposure_minutes": (float(r["exposure_minutes"])
                                 if r.get("exposure_minutes") not in (None, "")
                                 else None),
        })
    return out


def _amfmt(x: float) -> str:
    """Compact airmass label that always keeps a decimal point: 1.0, 1.3, 2.35."""
    s = f"{x:g}"
    return s if "." in s else s + ".0"


def pseudo_name(name: str, lo: float, hi: float) -> str:
    """GD71 + (1.0, 1.3) -> 'GD71@am1.0-1.3' (the queue entry's identity)."""
    return f"{name}@am{_amfmt(lo)}-{_amfmt(hi)}"


def build_pseudo_targets(standards: list[dict],
                         bins: list[tuple[float, float]],
                         priority: str = "P2",
                         exposure_minutes: float | None = None,
                         instrument: str = "LLAMAS") -> list[dict]:
    """One POST /v1/targets item per standard per airmass bin."""
    items = []
    for std in standards:
        exp = std.get("exposure_minutes") or exposure_minutes
        if exp is None or not math.isfinite(exp):
            raise ValueError(
                f"{std['name']}: no exposure — give --exposure-minutes or an "
                f"exposure_minutes column (the queue requires one)")
        for lo, hi in bins:
            items.append({
                "name": pseudo_name(std["name"], lo, hi),
                "ra": std["ra"], "dec": std["dec"],
                "mag": std.get("mag"),
                "priority": priority,
                "instrument": instrument,
                "exposure_minutes": float(exp),
                "airmass_min": lo, "airmass_max": hi,
                "notes": f"spectrophotometric standard {std['name']}, "
                         f"airmass bin {lo:g}-{hi:g}",
            })
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("standards", help="CSV/JSON of standards (name, ra, dec, mag)")
    ap.add_argument("--api", required=True, help="API base URL")
    ap.add_argument("--key", required=True,
                    help="bearer key — its program owns the pseudo-targets")
    ap.add_argument("--bins", default=DEFAULT_BINS,
                    help=f"comma-separated lo-hi airmass bins (default {DEFAULT_BINS})")
    ap.add_argument("--priority", default="P2", help="queue tier (default P2)")
    ap.add_argument("--instrument", default="LLAMAS")
    ap.add_argument("--exposure-minutes", type=float, default=None,
                    help="per-visit exposure when the CSV has no column")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the submission items without POSTing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    items = build_pseudo_targets(load_standards(args.standards),
                                 parse_bins(args.bins),
                                 priority=args.priority,
                                 exposure_minutes=args.exposure_minutes,
                                 instrument=args.instrument)
    if args.dry_run:
        print(json.dumps(items, indent=2))
        return 0

    req = urllib.request.Request(
        args.api.rstrip("/") + "/v1/targets",
        data=json.dumps(items).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {args.key}"},
        method="POST")
    with urllib.request.urlopen(req) as resp:
        results = json.loads(resp.read())
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    for r in results:
        if r.get("status") != "ok":
            logger.warning("rejected: %s", r)
    logger.info("enqueued %d/%d standard pseudo-targets", n_ok, len(items))
    print(json.dumps({"ok": n_ok, "total": len(items)}))
    return 0 if n_ok == len(items) else 1


if __name__ == "__main__":
    raise SystemExit(main())
