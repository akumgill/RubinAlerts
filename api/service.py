"""TargetQueueService — the framework-agnostic core of the submission API.

Holds the queue in memory and implements the six operations from the spec.
A web layer (api.app) is a thin adapter over this; tests drive it directly.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
from dataclasses import fields as _dc_fields
from typing import Callable, Optional

from .models import Target, TIERS, INSTRUMENTS

logger = logging.getLogger(__name__)

MATCH_ARCSEC = 2.0  # canonical-object dedup radius

# Target dataclass fields that are floats and use NaN-as-missing; stored as
# SQL NULL and restored to float('nan') on load.
_FLOAT_FIELDS = frozenset({
    "ra", "dec", "mag", "redshift", "exposure_minutes",
    "canonical_ra", "canonical_dec",
})
_COLUMNS = [f.name for f in _dc_fields(Target)]


class AuthError(Exception):
    """Unknown or missing API key."""


class NotFound(Exception):
    """Target id not found for this program."""


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    """Angular separation in arcsec (small-angle-safe haversine)."""
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    dd, dr = d2 - d1, r2 - r1
    a = math.sin(dd / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin(dr / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a)))) * 3600.0


def estimate_exposure_minutes(mag: float, redshift: float = float("nan")) -> float:
    """Queue-preview exposure estimate for the "requested hours" tally.

    Matches the scheduler's primary tier: the S/N ETC (Chris's LLAMAS SN Ia
    curve, binned S/N target, floored), so the preview and the actual plan agree.
    No moon factor here — the preview is night-agnostic; the scheduler applies
    moon. Falls back to the legacy magnitude scaling (mag 20 -> 45 min, 2.5x/mag)
    if the ETC can't produce a number."""
    if not (mag is not None and math.isfinite(mag)):
        return 45.0
    try:
        from core.snr_etc import (snr_exposure_minutes, MIN_EXPOSURE_MINUTES,
                                  MAX_EXPOSURE_MIN)
        t, _ = snr_exposure_minutes(mag)     # defaults: binned S/N=10, n_bin=10
        if math.isfinite(t):
            return float(min(MAX_EXPOSURE_MIN, max(MIN_EXPOSURE_MINUTES, t)))
    except Exception:
        pass
    return float(min(240.0, 45.0 * 2.5 ** (mag - 20.0)))


class TargetQueueService:
    def __init__(self, programs: dict, allocations_path: str,
                 resolver: Optional[Callable[[str], Optional[dict]]] = None,
                 db_path: Optional[str] = None):
        """
        programs : {api_key: program_name}
        allocations_path : YAML consumed by the orchestrator for budgets
        resolver : name -> {ra, dec, mag?, redshift?, scheme?} or None
        db_path : SQLite file path. If None, falls back to env ``DB_PATH``;
                  if that is also unset, an in-memory database is used (the
                  historical behavior — nothing is persisted). The deployment
                  layer (api.app) passes ``DB_PATH`` (default ./data/queue.db).
        """
        self.allocations_path = allocations_path
        self._resolver = resolver
        self._targets: list[Target] = []
        self._next_id = 1
        self._revision = 0   # bumped on every write; keys the dashboard cache
        self._lock = threading.RLock()

        # ---- open / init the backing store ----
        if db_path is None:
            db_path = os.environ.get("DB_PATH")
        self._db_path = db_path or ":memory:"
        if self._db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self._db_path))
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        # ---- program/keys config: constructor wins, else load from DB ----
        programs = dict(programs or {})
        if programs:
            self._programs = programs
            self._persist_config()
        else:
            self._programs = self._load_config()

        # ---- load persisted targets ----
        self._load_targets()

    # ---- persistence ----
    def _init_schema(self) -> None:
        cols = ", ".join(
            f"{c} INTEGER PRIMARY KEY" if c == "id" else f"{c}"
            for c in _COLUMNS
        )
        with self._conn:
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS targets ({cols})")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS config "
                "(key TEXT PRIMARY KEY, value TEXT)")

    def _persist_config(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO config(key, value) VALUES (?, ?)",
                ("programs", json.dumps(self._programs)))

    def _load_config(self) -> dict:
        row = self._conn.execute(
            "SELECT value FROM config WHERE key = 'programs'").fetchone()
        return json.loads(row["value"]) if row else {}

    @staticmethod
    def _row_to_target(row: sqlite3.Row) -> Target:
        kwargs = {}
        for c in _COLUMNS:
            v = row[c]
            if v is None and c in _FLOAT_FIELDS:
                v = float("nan")
            kwargs[c] = v
        return Target(**kwargs)

    def _persist(self, t: Target) -> None:
        vals = []
        for c in _COLUMNS:
            v = getattr(t, c)
            if isinstance(v, float) and math.isnan(v):
                v = None
            vals.append(v)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO targets ({', '.join(_COLUMNS)}) "
                f"VALUES ({placeholders})", vals)
            self._revision += 1   # invalidates the dashboard cache

    def _load_targets(self) -> None:
        rows = self._conn.execute(
            "SELECT * FROM targets ORDER BY id").fetchall()
        self._targets = [self._row_to_target(r) for r in rows]
        self._next_id = (max((t.id for t in self._targets), default=0) + 1)

    # ---- auth ----
    def program_for(self, api_key: str) -> str:
        prog = self._programs.get(api_key)
        if not prog:
            raise AuthError("unknown API key")
        return prog

    # ---- helpers ----
    def active(self, instrument: Optional[str] = None) -> list[Target]:
        """Active (queued/scheduled) targets. If ``instrument`` is given,
        restrict to that instrument's universe — its own targets plus any
        flagged EITHER — since LLAMAS and LDSS3 schedule as parallel systems."""
        act = [t for t in self._targets if t.status in ("queued", "scheduled")]
        if instrument:
            inst = instrument.upper()
            act = [t for t in act if t.instrument in (inst, "EITHER")]
        return act

    def _find_same_object(self, program: str, ra: float, dec: float
                          ) -> Optional[Target]:
        for t in self.active():
            if t.program == program and math.isfinite(t.canonical_ra):
                if _sep_arcsec(ra, dec, t.canonical_ra, t.canonical_dec) <= MATCH_ARCSEC:
                    return t
        return None

    def _shared_with(self, program: str, ra: float, dec: float) -> list[str]:
        others = set()
        for t in self.active():
            if t.program != program and math.isfinite(t.canonical_ra):
                if _sep_arcsec(ra, dec, t.canonical_ra, t.canonical_dec) <= MATCH_ARCSEC:
                    others.add(t.program)
        return sorted(others)

    # ---- operations ----
    def submit(self, api_key: str, items: list[dict]) -> list[dict]:
        """POST /v1/targets — upsert one or many. Returns a per-item result
        array so one bad target doesn't fail the batch."""
        program = self.program_for(api_key)
        results = []
        with self._lock:
            for raw in items:
                try:
                    results.append(self._submit_one(program, raw))
                except Exception as e:           # never let one item 500 the batch
                    results.append({"status": "error", "error": str(e),
                                    "submitted": raw})
        return results

    def revision(self) -> int:
        """Monotonic write counter — bumped on every submit/patch/withdraw. Keys
        the dashboard cache so it invalidates exactly when the queue changes."""
        return self._revision

    def has_targets(self) -> bool:
        """True if any targets are stored (used to gate demo seeding)."""
        return bool(self._targets)

    def _submit_one(self, program: str, raw: dict) -> dict:
        pri = str(raw.get("priority", "")).upper()
        if pri not in TIERS:
            return {"status": "error",
                    "error": f"priority must be one of {TIERS}"}
        inst = str(raw.get("instrument", "LLAMAS")).upper()
        if inst not in INSTRUMENTS:
            return {"status": "error",
                    "error": f"instrument must be one of {INSTRUMENTS}"}

        ra = raw.get("ra", float("nan"))
        dec = raw.get("dec", float("nan"))
        ra = float(ra) if ra is not None else float("nan")
        dec = float(dec) if dec is not None else float("nan")
        name = raw.get("name")
        mag = raw.get("mag", float("nan"))
        redshift = raw.get("redshift", float("nan"))
        resolved_from = "coords"

        # Resolve a name to coordinates when coords weren't given.
        if not (math.isfinite(ra) and math.isfinite(dec)):
            if name and self._resolver:
                hit = self._resolver(name)
                if not hit:
                    return {"status": "error", "name": name,
                            "error": "could not resolve identifier; send ra/dec"}
                ra, dec = float(hit["ra"]), float(hit["dec"])
                if mag in (None, "") or (isinstance(mag, float) and math.isnan(mag)):
                    mag = hit.get("mag", float("nan"))
                if redshift in (None, "") or (isinstance(redshift, float) and math.isnan(redshift)):
                    redshift = hit.get("redshift", float("nan"))
                resolved_from = f"name:{hit.get('scheme', 'unknown')}"
            else:
                return {"status": "error", "name": name,
                        "error": "no coordinates and no resolvable name"}

        mag = float(mag) if mag not in (None, "") else float("nan")
        redshift = float(redshift) if redshift not in (None, "") else float("nan")

        # A brand-new coords-only source we have no photometry for needs a mag
        # for exposure sizing.
        if resolved_from == "coords" and not math.isfinite(mag):
            return {"status": "error",
                    "error": "coords-only source needs an anticipated mag"}

        existing = self._find_same_object(program, ra, dec)
        if existing:  # upsert
            existing.priority = pri
            existing.instrument = inst
            if math.isfinite(mag):
                existing.mag = mag
            if math.isfinite(redshift):
                existing.redshift = redshift
            if raw.get("exposure_minutes") not in (None, ""):
                existing.exposure_minutes = float(raw["exposure_minutes"])
            if name:
                existing.name = name
            if raw.get("valid_until"):
                existing.valid_until = raw["valid_until"]
            t = existing
            updated = True
        else:
            t = Target(
                priority=pri, instrument=inst, name=name, ra=ra, dec=dec, mag=mag,
                band=str(raw.get("band", "r")), redshift=redshift,
                exposure_minutes=float(raw["exposure_minutes"])
                if raw.get("exposure_minutes") not in (None, "") else float("nan"),
                valid_until=raw.get("valid_until"),
                notes=str(raw.get("notes", "")),
                id=self._next_id, program=program, status="queued",
                canonical_ra=ra, canonical_dec=dec, resolved_from=resolved_from,
            )
            self._next_id += 1
            self._targets.append(t)
            updated = False

        self._persist(t)
        return {"status": "ok", "id": t.id, "updated": updated,
                "canonical_ra": round(ra, 5), "canonical_dec": round(dec, 5),
                "resolved_from": resolved_from,
                "shared_with": self._shared_with(program, ra, dec)}

    def list_targets(self, api_key: str) -> list[dict]:
        program = self.program_for(api_key)
        return [t.to_dict() for t in self._targets if t.program == program]

    def _owned(self, api_key: str, target_id: int) -> Target:
        program = self.program_for(api_key)
        for t in self._targets:
            if t.id == target_id and t.program == program:
                return t
        raise NotFound(f"target {target_id} not found for {program}")

    def patch(self, api_key: str, target_id: int, changes: dict) -> dict:
        t = self._owned(api_key, target_id)
        if "priority" in changes:
            pri = str(changes["priority"]).upper()
            if pri not in TIERS:
                raise ValueError(f"priority must be one of {TIERS}")
            t.priority = pri
        for k in ("mag", "redshift", "exposure_minutes"):
            if k in changes and changes[k] is not None:
                setattr(t, k, float(changes[k]))
        if "valid_until" in changes:
            t.valid_until = changes["valid_until"]
        self._persist(t)
        return t.to_dict()

    def withdraw(self, api_key: str, target_id: int) -> dict:
        t = self._owned(api_key, target_id)
        t.status = "withdrawn"
        self._persist(t)
        return {"status": "ok", "id": target_id, "new_status": "withdrawn"}

    def queue_summary(self) -> dict:
        """GET /v1/queue — counts per tier per program + requested hours."""
        by_prog: dict = {}
        for t in self.active():
            p = by_prog.setdefault(
                t.program, {"counts": {tier: 0 for tier in TIERS},
                            "requested_hours": 0.0})
            p["counts"][t.priority] += 1
            exp = (t.exposure_minutes if math.isfinite(t.exposure_minutes)
                   else estimate_exposure_minutes(t.mag, t.redshift))
            p["requested_hours"] += exp / 60.0
        for p in by_prog.values():
            p["requested_hours"] = round(p["requested_hours"], 2)
        return by_prog

    def plan_preview(self, date: str, moon: Optional[str] = None,
                     instrument: str = "LLAMAS") -> dict:
        """GET /v1/plan/preview — live dry-run over the current queue for one
        instrument (LLAMAS or LDSS3): only that instrument's targets are
        scheduled, with that instrument's overhead."""
        from .scheduler_bridge import preview_plan
        return preview_plan(self, date, moon, instrument)
