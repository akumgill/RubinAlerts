"""SelectionStore — persisted nightly SN Ia selection results.

The alert pipeline ranks candidates each night (nights/wide/ut{YYYYMMDD}/
candidates.csv); scripts/upload_selection_night.py POSTs a distilled copy to
POST /v1/selection/nights, which lands here. One row per night, keyed by the
ut-stamp; re-uploads upsert. Lives in the same SQLite file as the target queue
(api.service.TargetQueueService) but is a separate table with its own
connection — nothing here touches the dashboard cache.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

NIGHT_STAMP_RE = re.compile(r"^ut\d{8}$")
MAX_CANDIDATES = 200           # per-night row cap enforced at ingest
# Serialized-payload cap. Raised 1 MB -> 2.5 MB (2026-08-18) when compact
# per-candidate light curves (`lc` rows, <=150 points) joined the payload:
# a full night with lc measures ~200-400 KB, so 2.5 MB keeps generous headroom
# while still bounding a runaway upload.
MAX_PAYLOAD_BYTES = 2_500_000


class SelectionStore:
    """Nightly selection results in a `selection_nights` table.

    Schema: night_stamp TEXT PRIMARY KEY (e.g. 'ut20260818'), mjd REAL,
    uploaded_at TEXT (UTC ISO), summary_json TEXT, candidates_json TEXT.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.environ.get("DB_PATH")
        self._db_path = db_path or ":memory:"
        if self._db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self._db_path))
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS selection_nights ("
                "night_stamp TEXT PRIMARY KEY, mjd REAL, uploaded_at TEXT, "
                "summary_json TEXT, candidates_json TEXT)")

    # ---- writes ----
    def upsert_night(self, night_stamp: str, mjd: Optional[float],
                     summary: dict, candidates: list[dict]) -> dict:
        """Insert or replace one night's selection. Candidates are stored in
        the order given, which is rank order (descending merit)."""
        if not NIGHT_STAMP_RE.match(night_stamp or ""):
            raise ValueError("night_stamp must match ut{YYYYMMDD}")
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list")
        if len(candidates) > MAX_CANDIDATES:
            raise ValueError(f"too many candidates (max {MAX_CANDIDATES})")
        summary = dict(summary or {})
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO selection_nights "
                "(night_stamp, mjd, uploaded_at, summary_json, candidates_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (night_stamp, mjd, now, json.dumps(summary),
                 json.dumps(candidates)))
        logger.info("selection night %s stored (%d candidates)",
                    night_stamp, len(candidates))
        return {"ok": True, "night_stamp": night_stamp,
                "n_candidates": len(candidates)}

    # ---- reads ----
    def fetch_nights(self, limit: int = 10) -> list[dict]:
        """Most-recent-first list of stored nights (the ut-stamp sorts
        lexically == chronologically)."""
        limit = max(1, min(int(limit), 100))
        rows = self._conn.execute(
            "SELECT * FROM selection_nights ORDER BY night_stamp DESC LIMIT ?",
            (limit,)).fetchall()
        return [{
            "night_stamp": r["night_stamp"],
            "mjd": r["mjd"],
            "uploaded_at": r["uploaded_at"],
            "summary": json.loads(r["summary_json"] or "{}"),
            "candidates": json.loads(r["candidates_json"] or "[]"),
        } for r in rows]


def candidate_key(cand: dict) -> Optional[str]:
    """Stable cross-night identity: ztf_oid when present, else diaObjectId."""
    for k in ("ztf_oid", "diaObjectId"):
        v = cand.get(k)
        if v not in (None, "", "nan"):
            return str(v)
    return None


def compute_persistence(nights: list[dict]) -> list[dict]:
    """Objects appearing in >= 2 of the given nights, with their per-night
    rank (1-based position in the stored order) and ranking value. The
    'merit' field carries score_rate when the night has it (the PI-approved
    2026-08-18 ordering) and falls back to the legacy merit on old nights —
    matching how the stored order itself was produced.

    ``nights`` is newest-first (as fetch_nights returns). Each result row:
      {id, label, tns_name, appearances: [{night_stamp, rank, merit}, ...
       oldest->newest], latest_merit, latest_rank}
    Sorted by best (lowest) most-recent rank.
    """
    by_id: dict[str, dict] = {}
    for night in reversed(nights):                 # oldest -> newest
        stamp = night.get("night_stamp")
        for i, cand in enumerate(night.get("candidates") or []):
            key = candidate_key(cand)
            if key is None:
                continue
            rec = by_id.setdefault(key, {
                "id": key, "label": key, "tns_name": None, "appearances": []})
            value = cand.get("score_rate")
            if value is None:
                value = cand.get("merit")
            rec["appearances"].append({
                "night_stamp": stamp, "rank": i + 1,
                "merit": value})
            # latest non-null naming wins
            tns = cand.get("tns_name")
            if tns not in (None, "", "nan"):
                rec["tns_name"] = tns
            for name_key in ("tns_name", "ztf_oid", "diaObjectId"):
                v = cand.get(name_key)
                if v not in (None, "", "nan"):
                    rec["label"] = str(v)
                    break
    persistent = []
    for rec in by_id.values():
        if len(rec["appearances"]) < 2:
            continue
        last = rec["appearances"][-1]
        rec["latest_rank"] = last["rank"]
        rec["latest_merit"] = last["merit"]
        persistent.append(rec)
    persistent.sort(key=lambda r: (r["latest_rank"], r["id"]))
    return persistent
