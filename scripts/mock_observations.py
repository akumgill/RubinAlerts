#!/usr/bin/env python
"""Mock observation-night generator (the FITS source is not real yet).

Writes synthetic FITS files with minimal headers matching the adapter's
assumed dialect (scripts/ingest_fits_night.py), simulating a plausible LLAMAS
night against the CURRENT queue in a SQLite DB:

  * the top 4 science targets (non-standards, LLAMAS/EITHER, by tier then id)
    each observed as a CR triplet: 3 x (total exposure / 3),
  * ONE airmass-binned standard observed in TWO of its bins (exposure taken
    at a time when its computed airmass actually falls inside each bin),
  * ONE unassociated pointing (offset > 1 arcmin from any queue target).

Used by the tests and to seed the local demo DB:
  python scripts/mock_observations.py --db ./data/queue.db \
      --date 2026-08-16 --out /tmp/mock_night
then ingest with scripts/ingest_fits_night.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

logger = logging.getLogger(__name__)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _fmt_hms(ra_deg: float) -> str:
    h = ra_deg / 15.0
    H = int(h); m = (h - H) * 60; M = int(m); S = (m - M) * 60
    return f"{H:02d}:{M:02d}:{S:05.2f}"


def _fmt_dms(dec_deg: float) -> str:
    sign = "-" if dec_deg < 0 else "+"
    d = abs(dec_deg)
    D = int(d); m = (d - D) * 60; M = int(m); S = (m - M) * 60
    return f"{sign}{D:02d}:{M:02d}:{S:04.1f}"


def _write_fits(path, name, ra, dec, utc, exptime, instrument="LLAMAS"):
    from astropy.io import fits as pyfits
    hdu = pyfits.PrimaryHDU()
    hdu.header["OBJECT"] = name
    hdu.header["RA"] = _fmt_hms(ra)          # sexagesimal, RA in HOURS
    hdu.header["DEC"] = _fmt_dms(dec)
    hdu.header["EXPTIME"] = float(exptime)
    hdu.header["DATE-OBS"] = utc
    hdu.header["INSTRUME"] = instrument
    hdu.writeto(path, overwrite=True)
    return path


def _triplet_seconds(t) -> float:
    """Per-sub-exposure seconds: the target's own triplet spec if set, else
    total/3 rounded to 10 s (the canonical CR protocol)."""
    if t.n_exposures and math.isfinite(t.exposure_seconds):
        return float(t.exposure_seconds)
    total_s = (t.exposure_minutes if math.isfinite(t.exposure_minutes)
               else 45.0) * 60.0
    return max(10.0, round(total_s / 3.0 / 10.0) * 10.0)


def _time_in_bin(ra, dec, date, lo, hi):
    """First UT time in the night when (ra, dec) sits inside airmass
    [lo, hi], sampled every 15 min over 00:30-09:45 UT. None if never."""
    from api.observations import airmass_at
    for step in range(38):
        utc = _utc(date, 0.5 + step * 0.25)
        am = airmass_at(ra, dec, utc)
        if am is not None and lo <= am <= hi:
            return utc, am
    return None, None


def _utc(date: str, hours: float) -> str:
    h = int(hours); m = int(round((hours - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{date}T{h:02d}:{m:02d}:00"



class _RemoteTarget:
    """Minimal stand-in for a queue Target, built from a /v1/targets row.

    The mock generator decides what to shoot by reading a LOCAL queue, which
    means seeding a remote instance used to need the same night staged twice —
    once there, once locally — with the two silently free to drift apart. This
    reads the remote queue directly, so the frames are generated against the
    exact rows the ingest will associate them with.
    """

    __slots__ = ("id", "name", "priority", "canonical_ra", "canonical_dec",
                 "exposure_minutes", "n_exposures", "exposure_seconds",
                 "airmass_min", "airmass_max", "instrument", "status")

    def __init__(self, row: dict):
        def _f(k):
            v = row.get(k)
            return float(v) if isinstance(v, (int, float)) else float("nan")
        self.id = row.get("id") or 0
        self.name = row.get("name") or ""
        self.priority = str(row.get("priority") or "P3")
        # canonical_* is the resolved position and the key the ingest
        # associates on; fall back to the submitted coords for older rows.
        self.canonical_ra = _f("canonical_ra")
        self.canonical_dec = _f("canonical_dec")
        if not math.isfinite(self.canonical_ra):
            self.canonical_ra = _f("ra")
        if not math.isfinite(self.canonical_dec):
            self.canonical_dec = _f("dec")
        self.exposure_minutes = _f("exposure_minutes")
        n = row.get("n_exposures")
        self.n_exposures = int(n) if isinstance(n, int) else None
        self.exposure_seconds = _f("exposure_seconds")
        self.airmass_min = _f("airmass_min")
        self.airmass_max = _f("airmass_max")
        self.instrument = str(row.get("instrument") or "LLAMAS").upper()
        self.status = str(row.get("status") or "queued")


def targets_from_api(api: str, key: str, instrument: str = "LLAMAS") -> list:
    """Live queue rows for ``instrument`` (plus EITHER), still queued."""
    import urllib.request
    req = urllib.request.Request(
        api.rstrip("/") + "/v1/targets",
        headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req) as r:
        rows = json.loads(r.read())
    out = []
    for row in rows:
        t = _RemoteTarget(row)
        if t.status not in ("queued", "scheduled"):
            continue
        if t.instrument not in (instrument.upper(), "EITHER"):
            continue
        if not (math.isfinite(t.canonical_ra) and math.isfinite(t.canonical_dec)):
            logger.warning("%s has no usable coordinates; skipped", t.name)
            continue
        out.append(t)
    logger.info("read %d live target(s) from %s", len(out), api)
    return out


def generate(db_path: str = None, date: str = None, out_dir: str = None,
             n_targets: int = 4, api: str = None, key: str = None) -> list:
    """Write the mock night's FITS files; returns the written paths.

    Targets come from a local SQLite queue (``db_path``) or, when ``api`` and
    ``key`` are given, straight from a live instance — which is what you want
    when seeding a remote site, since the frames must be generated against the
    same rows the ingest will associate them with.
    """
    os.makedirs(out_dir, exist_ok=True)
    if api and key:
        active = targets_from_api(api, key)
    else:
        from api.service import TargetQueueService
        svc = TargetQueueService({}, "ref/allocations_LLAMAS_2026B.yaml",
                                 db_path=db_path)
        active = svc.active(instrument="LLAMAS")
    is_std = lambda t: (math.isfinite(t.airmass_min)
                        or math.isfinite(t.airmass_max))
    science = sorted([t for t in active if not is_std(t)],
                     key=lambda t: (t.priority, t.id))[:n_targets]
    standards = [t for t in active if is_std(t)]

    written, seq = [], 0
    hour = 1.0                                   # cursor through the night (UT)

    def shoot(name, ra, dec, utc, exptime):
        nonlocal seq
        seq += 1
        path = os.path.join(out_dir, f"llamas_{date.replace('-', '')}_"
                                     f"{seq:04d}.fits")
        written.append(_write_fits(path, name, ra, dec, utc, exptime))

    # top-4 science targets, each a CR triplet of total/3
    for t in science:
        sub_s = _triplet_seconds(t)
        for k in range(3):
            shoot(t.name, t.canonical_ra, t.canonical_dec,
                  _utc(date, hour), sub_s)
            hour += sub_s / 3600.0 + 0.02
        hour += 0.1

    # one standard in two airmass bins (two pseudo-targets share coords)
    std_by_star: dict = {}
    for t in standards:
        star = (t.name or "").split("@")[0]
        std_by_star.setdefault(star, []).append(t)
    for star, bins in std_by_star.items():
        if len(bins) < 2:
            continue
        n_done = 0
        for t in sorted(bins, key=lambda x: x.airmass_min):
            lo = t.airmass_min if math.isfinite(t.airmass_min) else 1.0
            hi = t.airmass_max if math.isfinite(t.airmass_max) else 3.0
            utc, am = _time_in_bin(t.canonical_ra, t.canonical_dec, date, lo, hi)
            if utc is None:
                logger.warning("standard %s: no time tonight inside airmass "
                               "%.2f-%.2f; bin skipped", t.name, lo, hi)
                continue
            shoot(star, t.canonical_ra, t.canonical_dec, utc,
                  (t.exposure_minutes if math.isfinite(t.exposure_minutes)
                   else 6.0) * 60.0)
            n_done += 1
            if n_done == 2:
                break
        if n_done:
            break                                # one standard is enough

    # one unassociated pointing: offset > 1 arcmin from the first target
    if science:
        base = science[0]
        shoot("RANDOM-PTG", base.canonical_ra,
              base.canonical_dec + 2.5 / 60.0,   # +2.5 arcmin in dec
              _utc(date, hour + 0.2), 120.0)

    logger.info("mock night: %d FITS files in %s", len(written), out_dir)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", help="queue SQLite file (local)")
    src.add_argument("--api", help="live API base URL — read its queue instead")
    ap.add_argument("--key", help="bearer key, required with --api")
    ap.add_argument("--date", required=True, help="UT date, e.g. 2026-08-16")
    ap.add_argument("--out", required=True, help="output directory for FITS")
    ap.add_argument("--n-targets", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.api and not args.key:
        ap.error("--key is required with --api")
    files = generate(db_path=args.db, date=args.date, out_dir=args.out,
                     n_targets=args.n_targets, api=args.api, key=args.key)
    print(f"{len(files)} files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
