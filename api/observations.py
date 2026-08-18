"""ObservationStore — ingested observations (what was ACTUALLY shot on sky).

The real FITS delivery location/dialect is still unknown, so everything here
works on a CANONICAL record; only the thin header adapter
(scripts/ingest_fits_night.py) is expected to change when we learn the real
dialect. Records arrive via POST /v1/observations (or the adapter's --db
mode) and the SERVER does the association against the target queue:

  1. POINTING: canonical target coordinates within 1 arcmin (Chris's rule).
     Standards pseudo-targets sharing coordinates across airmass bins are
     disambiguated by the airmass AT utc_start (computed from ra/dec + LCO)
     against each bin's [airmass_min, airmass_max].
  2. NAME fallback: case-insensitive, whitespace-stripped match.
  3. else UNASSOCIATED.

Accounting: each associated observation's exptime is charged to the owning
program — split EVENLY when the same coordinates were enqueued by more than
one program (Chris's rule) — into an `observation_charges` table keyed by
(filename, program), so re-ingesting a night REPLACES its charges instead of
double-charging (idempotent by filename). The dashboard burndown folds these
charges into the allocations overview's `used`.

Same SQLite file as the queue/selection tables; own connection; independent
of the dashboard cache (the API layer re-reads charges per request).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
from typing import Optional

logger = logging.getLogger(__name__)

MATCH_POINTING_ARCSEC = 60.0    # Chris: associate by pointing within 1 arcmin
SAME_COORD_ARCSEC = 2.0         # "same coordinates" for the even-split rule

_UT_RE = re.compile(r"^ut(\d{8})$")


def night_stamp_of(utc_start: str) -> str:
    """utYYYYMMDD from an ISO utc_start. ASSUMPTION: the UT date of the
    exposure IS the night stamp (LCO nights span one UT date; revisit if the
    real delivery groups differently)."""
    digits = re.sub(r"[^0-9]", "", (utc_start or "")[:10])
    return f"ut{digits}" if len(digits) == 8 else "ut00000000"


def normalize_night(night: str) -> str:
    """'2026-08-18' or 'ut20260818' -> 'ut20260818'."""
    night = (night or "").strip()
    if _UT_RE.match(night):
        return night
    digits = re.sub(r"[^0-9]", "", night)
    return f"ut{digits}" if len(digits) == 8 else night


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    dd, dr = d2 - d1, r2 - r1
    a = math.sin(dd / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin(dr / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a)))) * 3600.0


def airmass_at(ra: float, dec: float, utc_start: str) -> Optional[float]:
    """Airmass of (ra, dec) at ``utc_start`` from Las Campanas (sec-z form,
    matching the planner). None when below the horizon or uncomputable."""
    try:
        from astropy.coordinates import SkyCoord, EarthLocation, AltAz
        from astropy.time import Time
        import astropy.units as u
        from config import OBSERVATORY_CONFIG as OC
        site = EarthLocation(lat=OC.latitude * u.deg, lon=OC.longitude * u.deg,
                             height=OC.elevation_m * u.m)
        alt = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).transform_to(
            AltAz(obstime=Time(utc_start), location=site)).alt.deg
        if alt <= 0:
            return None
        return float(1.0 / math.sin(math.radians(alt)))
    except Exception as e:
        logger.warning("airmass computation failed for %s: %s", utc_start, e)
        return None


def _norm_name(name) -> str:
    return re.sub(r"\s+", "", str(name or "")).lower()


def _in_bin(t, am: Optional[float]) -> bool:
    """True when ``am`` sits inside the target's airmass range (open bounds
    pass automatically; unknown airmass never satisfies a constrained bin)."""
    lo = getattr(t, "airmass_min", float("nan"))
    hi = getattr(t, "airmass_max", float("nan"))
    has_lo, has_hi = math.isfinite(lo), math.isfinite(hi)
    if not has_lo and not has_hi:
        return True
    if am is None:
        return False
    return (not has_lo or am >= lo) and (not has_hi or am <= hi)


def associate(targets: list, rec: dict, airmass: Optional[float]) -> dict:
    """Match one canonical record against the queue targets.

    Returns {method, target, programs}: ``target`` is the primary match (for
    target_id), ``programs`` every program owed a share of the time (the
    even-split rule when the same coordinates were enqueued twice).
    """
    ra, dec = rec.get("ra"), rec.get("dec")
    coords_ok = (ra is not None and dec is not None
                 and math.isfinite(float(ra)) and math.isfinite(float(dec)))
    if coords_ok:
        hits = [t for t in targets
                if math.isfinite(t.canonical_ra)
                and _sep_arcsec(float(ra), float(dec),
                                t.canonical_ra, t.canonical_dec)
                <= MATCH_POINTING_ARCSEC]
        if hits:
            # standards disambiguation: among airmass-binned pseudo-targets,
            # keep the bin(s) containing the airmass at utc_start
            binned = [t for t in hits
                      if math.isfinite(getattr(t, "airmass_min", float("nan")))
                      or math.isfinite(getattr(t, "airmass_max", float("nan")))]
            if binned:
                in_bin = [t for t in binned if _in_bin(t, airmass)]
                unbinned = [t for t in hits if t not in binned]
                hits = (in_bin + unbinned) if in_bin else (unbinned or hits)
            primary = min(hits, key=lambda t: _sep_arcsec(
                float(ra), float(dec), t.canonical_ra, t.canonical_dec))
            # even split across programs that enqueued the SAME coordinates
            programs = sorted({t.program for t in hits
                               if _sep_arcsec(t.canonical_ra, t.canonical_dec,
                                              primary.canonical_ra,
                                              primary.canonical_dec)
                               <= SAME_COORD_ARCSEC})
            return {"method": "pointing", "target": primary,
                    "programs": programs or [primary.program]}
    raw = _norm_name(rec.get("object_name_raw"))
    if raw:
        for t in targets:
            if _norm_name(t.name) == raw:
                return {"method": "name", "target": t, "programs": [t.program]}
    return {"method": "unassociated", "target": None, "programs": []}


class ObservationStore:
    """`observations` + `observation_charges` tables (same DB file as the
    queue; own connection, following SelectionStore's pattern)."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.environ.get("DB_PATH")
        self._db_path = db_path or ":memory:"
        if self._db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self._db_path)),
                        exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS observations ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, utc_start TEXT, "
                "mjd REAL, ra REAL, dec REAL, object_name_raw TEXT, "
                "exptime_s REAL, instrument TEXT, filename TEXT UNIQUE, "
                "night_stamp TEXT, program TEXT, target_id INTEGER, "
                "target_name TEXT, assoc_method TEXT, airmass REAL)")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS observation_charges ("
                "filename TEXT, program TEXT, seconds REAL, instrument TEXT, "
                "night_stamp TEXT, PRIMARY KEY (filename, program))")

    # ---- ingest ----
    def ingest(self, targets: list, records: list[dict]) -> list[dict]:
        """Associate + store a batch of canonical records; returns per-record
        outcomes. Idempotent by filename (row upserted, charges replaced)."""
        results = []
        for rec in records:
            try:
                results.append(self._ingest_one(targets, rec))
            except Exception as e:           # one bad record never kills a night
                logger.exception("observation ingest failed: %s", e)
                results.append({"filename": rec.get("filename"),
                                "status": "error", "error": str(e)})
        return results

    def _ingest_one(self, targets: list, rec: dict) -> dict:
        utc = str(rec.get("utc_start") or "")
        filename = rec.get("filename") or f"noname_{utc}_{rec.get('ra')}_{rec.get('dec')}"
        night = normalize_night(rec.get("night_stamp") or night_stamp_of(utc))
        ra = rec.get("ra")
        dec = rec.get("dec")
        exptime = float(rec.get("exptime_s") or 0.0)
        am = rec.get("airmass")
        if am is None and ra is not None and dec is not None and utc:
            am = airmass_at(float(ra), float(dec), utc)
        assoc = associate(targets, rec, am)
        t = assoc["target"]
        programs = assoc["programs"]
        with self._lock, self._conn:
            # dedupe: filename is UNIQUE; a filename-less record falls back to
            # (utc_start, ra, dec) via the synthesized filename above
            self._conn.execute(
                "INSERT INTO observations (utc_start, mjd, ra, dec, "
                "object_name_raw, exptime_s, instrument, filename, "
                "night_stamp, program, target_id, target_name, assoc_method, "
                "airmass) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(filename) DO UPDATE SET "
                "utc_start=excluded.utc_start, mjd=excluded.mjd, "
                "ra=excluded.ra, dec=excluded.dec, "
                "object_name_raw=excluded.object_name_raw, "
                "exptime_s=excluded.exptime_s, instrument=excluded.instrument, "
                "night_stamp=excluded.night_stamp, program=excluded.program, "
                "target_id=excluded.target_id, target_name=excluded.target_name, "
                "assoc_method=excluded.assoc_method, airmass=excluded.airmass",
                (utc, rec.get("mjd"), ra, dec, rec.get("object_name_raw"),
                 exptime, rec.get("instrument"), filename, night,
                 (programs[0] if len(programs) == 1 else
                  ("+".join(programs) if programs else None)),
                 (t.id if t is not None else None),
                 (t.name if t is not None else None),
                 assoc["method"], am))
            # charges: REPLACE this filename's charges (idempotent re-ingest);
            # even split across programs sharing the coordinates
            self._conn.execute(
                "DELETE FROM observation_charges WHERE filename = ?", (filename,))
            if programs and exptime > 0:
                share = exptime / len(programs)
                for prog in programs:
                    self._conn.execute(
                        "INSERT INTO observation_charges "
                        "(filename, program, seconds, instrument, night_stamp) "
                        "VALUES (?,?,?,?,?)",
                        (filename, prog, share,
                         rec.get("instrument") or "LLAMAS", night))
        return {"filename": filename, "night_stamp": night,
                "assoc_method": assoc["method"],
                "target_id": t.id if t is not None else None,
                "target_name": t.name if t is not None else None,
                "programs": programs,
                "airmass": None if am is None else round(am, 3)}

    # ---- reads ----
    def night_rows(self, night: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM observations WHERE night_stamp = ? "
            "ORDER BY utc_start", (normalize_night(night),)).fetchall()
        return [dict(r) for r in rows]

    def observed_coords(self, limit: int = 1000) -> list[dict]:
        """Compact all-time pointing list (newest first) for 'observed'
        badges — no per-exposure history."""
        rows = self._conn.execute(
            "SELECT ra, dec, night_stamp, target_id, target_name, "
            "MAX(utc_start) AS last_utc, COUNT(*) AS n "
            "FROM observations GROUP BY ra, dec, night_stamp "
            "ORDER BY last_utc DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def used_hours_by_program(self) -> dict:
        """{program: {instrument: hours}} from ingested observation charges —
        what the dashboard burndown's `used` reads."""
        rows = self._conn.execute(
            "SELECT program, COALESCE(instrument, 'LLAMAS') AS inst, "
            "SUM(seconds) AS s FROM observation_charges "
            "GROUP BY program, inst").fetchall()
        out: dict = {}
        for r in rows:
            out.setdefault(r["program"], {})[r["inst"]] = round(r["s"] / 3600.0, 3)
        return out

    def observed_target_ids(self) -> dict:
        """{target_id: latest night_stamp} for queue 'observed' markers."""
        rows = self._conn.execute(
            "SELECT target_id, MAX(night_stamp) AS night FROM observations "
            "WHERE target_id IS NOT NULL GROUP BY target_id").fetchall()
        return {r["target_id"]: r["night"] for r in rows}
