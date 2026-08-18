#!/usr/bin/env python
"""Ingest a night of FITS files into the observations repository.

Walks a directory of FITS files, adapts their headers into the CANONICAL
observation record (api/observations.py), and either POSTs them to
POST /v1/observations (--api + --key) or ingests straight into the SQLite
store (--db, local dev — runs the same server-side association).

=============================================================================
HEADER-FIELD MAPPING — THE ONLY PART EXPECTED TO CHANGE (real dialect TBD).
The actual LCO/Magellan delivery location and header dialect are unknown;
`record_from_header` below encodes our current ASSUMPTIONS:
  OBJECT    -> object_name_raw   (also tries OBJNAME, TARGNAME)
  RA / DEC  -> pointing; sexagesimal (HH:MM:SS / +DD:MM:SS, RA in HOURS) or
               decimal degrees both tolerated (also tries RA-D/DEC-D, TELRA/
               TELDEC)
  EXPTIME   -> exptime_s         (also tries EXPOSURE, ITIME) — seconds
  DATE-OBS  -> utc_start; ISO 'YYYY-MM-DDThh:mm:ss[.s]' (also tries UT-DATE +
               UT-TIME pair)
  INSTRUME  -> instrument        (also tries INSTRUMENT); default LLAMAS
Everything downstream (association, accounting, dashboard) consumes only the
canonical record and needs no change when the dialect lands.
=============================================================================

Usage:
  python scripts/ingest_fits_night.py /path/to/fits_dir \
      --api http://localhost:8000 --key <bearer>
  python scripts/ingest_fits_night.py /path/to/fits_dir --db ./data/queue.db
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import sys
import urllib.request

logger = logging.getLogger(__name__)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _first(header, *keys):
    for k in keys:
        if k in header and header[k] not in (None, ""):
            return header[k]
    return None


def _parse_angle(value, is_ra: bool) -> float:
    """Sexagesimal (RA in HOURS) or decimal-degree header angle -> degrees."""
    if value is None:
        return float("nan")
    s = str(value).strip()
    if re.match(r"^[+-]?\d+(\.\d+)?$", s):      # decimal degrees
        return float(s)
    parts = re.split(r"[:\s]+", s)
    if len(parts) >= 2:
        sign = -1.0 if parts[0].lstrip().startswith("-") else 1.0
        nums = [abs(float(p)) for p in parts[:3]] + [0.0] * (3 - len(parts[:3]))
        deg = nums[0] + nums[1] / 60.0 + nums[2] / 3600.0
        deg *= sign
        return deg * 15.0 if is_ra else deg
    return float("nan")


def record_from_header(header, filename: str) -> dict:
    """FITS header -> canonical observation record (see module docstring for
    the assumed dialect — this function is the swap point)."""
    ra = _parse_angle(_first(header, "RA", "RA-D", "TELRA"), is_ra=True)
    dec = _parse_angle(_first(header, "DEC", "DEC-D", "TELDEC"), is_ra=False)
    utc = _first(header, "DATE-OBS", "DATE_OBS")
    if utc is None:
        d, t = _first(header, "UT-DATE"), _first(header, "UT-TIME")
        utc = f"{d}T{t}" if d and t else ""
    exptime = _first(header, "EXPTIME", "EXPOSURE", "ITIME")
    mjd = None
    try:
        from astropy.time import Time
        mjd = float(Time(str(utc)).mjd) if utc else None
    except Exception:
        pass
    return {
        "utc_start": str(utc or ""),
        "mjd": mjd,
        "ra": None if not math.isfinite(ra) else ra,
        "dec": None if not math.isfinite(dec) else dec,
        "object_name_raw": str(_first(header, "OBJECT", "OBJNAME",
                                      "TARGNAME") or ""),
        "exptime_s": float(exptime) if exptime is not None else 0.0,
        "instrument": str(_first(header, "INSTRUME", "INSTRUMENT") or "LLAMAS"),
        "filename": os.path.basename(filename),
    }


def load_night_dir(fits_dir: str) -> list[dict]:
    from astropy.io import fits as pyfits
    paths = sorted(glob.glob(os.path.join(fits_dir, "*.fits"))
                   + glob.glob(os.path.join(fits_dir, "*.fits.gz")))
    records = []
    for p in paths:
        try:
            with pyfits.open(p) as hdul:
                records.append(record_from_header(hdul[0].header, p))
        except Exception as e:
            logger.warning("skipping unreadable FITS %s: %s", p, e)
    logger.info("adapted %d/%d FITS files from %s",
                len(records), len(paths), fits_dir)
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fits_dir", help="directory of the night's FITS files")
    ap.add_argument("--api", help="API base URL")
    ap.add_argument("--key", help="bearer API key for --api mode")
    ap.add_argument("--db", help="ingest straight into this SQLite file (dev)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.db and not (args.api and args.key):
        ap.error("give either --db PATH, or --api URL with --key KEY")

    records = load_night_dir(args.fits_dir)
    if not records:
        logger.error("no ingestible FITS files found")
        return 1

    if args.db:
        sys.path.insert(0, _REPO)
        from api.observations import ObservationStore
        from api.service import TargetQueueService
        svc = TargetQueueService({}, "ref/allocations_LLAMAS_2026B.yaml",
                                 db_path=args.db)
        store = ObservationStore(db_path=args.db)
        results = store.ingest(svc._targets, records)
        out = {"results": results, "n_records": len(results)}
    else:
        req = urllib.request.Request(
            args.api.rstrip("/") + "/v1/observations",
            data=json.dumps({"observations": records},
                            allow_nan=False).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {args.key}"},
            method="POST")
        with urllib.request.urlopen(req) as resp:
            out = json.loads(resp.read())

    by_method: dict = {}
    for r in out["results"]:
        by_method[r.get("assoc_method", "error")] = \
            by_method.get(r.get("assoc_method", "error"), 0) + 1
    logger.info("ingested %d records: %s", len(out["results"]), by_method)
    print(json.dumps(by_method))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
