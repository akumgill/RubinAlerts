#!/usr/bin/env python
"""Upload one night's ranked SN Ia candidates to the selection API.

Reads nights/<mode>/ut{YYYYMMDD}/candidates.csv (the alert pipeline's output),
distills the per-candidate fields the web selection page renders, builds the
night summary, and either POSTs it to /v1/selection/nights (--api + --key) or
writes it straight into the SQLite store (--db, for local dev — the deployed
server has no nights/ directory, so production always goes over HTTP).

Usage:
  python scripts/upload_selection_night.py nights/wide/ut20260818 \
      --api http://localhost:8000 --key <bearer-key>
  python scripts/upload_selection_night.py nights/wide/ut20260818 \
      --db ./data/queue.db

NaN-safe: every numeric field is finite or null (never a NaN literal in JSON).
The CSV schema has drifted across nights (e.g. ut20260713 predates ztf_oid,
z_source, salt_t0_err, w_iaspec); missing columns simply come out null.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from datetime import date
from statistics import median

logger = logging.getLogger(__name__)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIGHT_STAMP_RE = re.compile(r"^ut(\d{4})(\d{2})(\d{2})$")
MAX_CANDIDATES = 200
_MISSING = ("", "nan", "none", "null", "n/a")


@dataclass
class NightPayload:
    night_stamp: str
    mjd: float
    summary: dict
    candidates: list

    def to_json(self) -> str:
        return json.dumps({
            "night_stamp": self.night_stamp, "mjd": self.mjd,
            "summary": self.summary, "candidates": self.candidates,
        }, allow_nan=False)


def _s(row: dict, col: str) -> str | None:
    """String field: '' / 'nan' / missing column -> None."""
    v = (row.get(col) or "").strip()
    return None if v.lower() in _MISSING else v


def _f(row: dict, col: str) -> float | None:
    """Float field: unparseable / non-finite / missing -> None."""
    v = (row.get(col) or "").strip()
    if v.lower() in _MISSING:
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _i(row: dict, col: str) -> int | None:
    x = _f(row, col)
    return int(x) if x is not None else None


def _b(row: dict, col: str) -> bool | None:
    v = _s(row, col)
    if v is None:
        return None
    return v.lower() == "true"


def night_stamp_to_mjd(night_stamp: str) -> float:
    m = NIGHT_STAMP_RE.match(night_stamp)
    if not m:
        raise ValueError(f"night dir must be named ut{{YYYYMMDD}}, got {night_stamp!r}")
    d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return float((d - date(1858, 11, 17)).days)   # MJD epoch


def parse_candidate(row: dict) -> dict:
    """One CSV row -> the distilled candidate record the page renders."""
    return {
        "diaObjectId": _s(row, "diaObjectId"),
        "ztf_oid": _s(row, "ztf_oid"),
        "tns_name": _s(row, "tns_name"),
        "tns_type": _s(row, "tns_type"),
        "ra": _f(row, "ra"),
        "dec": _f(row, "dec"),
        "peak_mag": _f(row, "peak_mag"),
        # no latest/current-mag column exists in any night so far; probe the
        # plausible names so it fills in if the pipeline starts writing one
        "latest_mag": next((x for c in ("latest_mag", "current_mag", "mag_now")
                            if (x := _f(row, c)) is not None), None),
        "delta_t": _f(row, "delta_t"),
        # z_best is the fit-adopted redshift on newer nights; z provenance
        # lives in z_source (newer) with ned_name as the legacy carrier
        "z": _f(row, "z_best") if _f(row, "z_best") is not None else _f(row, "redshift"),
        "z_source": _s(row, "z_source") or _s(row, "ned_name"),
        # SALT2 free-z fit: a light-curve-based z ESTIMATE for candidates with
        # no spec/host redshift (displayed as "~z", never used as a real z)
        "salt_z": _f(row, "salt_z"),
        "salt_z_railed": _b(row, "salt_z_railed"),
        "merit": _f(row, "merit"),
        # PI-approved score (2026-08-18): score = P x V(z) x G x U x E, ranked
        # by score_rate = score x (45min/exp)^alpha. Old CSVs lack these -> null.
        "score": _f(row, "score"),
        "score_rate": _f(row, "score_rate"),
        "score_rank": _i(row, "score_rank"),
        "p_usable": _f(row, "p_usable"),
        "v_z": _f(row, "v_z"),
        "g_info": _f(row, "g_info"),
        "u_urgency": _f(row, "u_urgency"),
        "e_east": _f(row, "e_east"),
        "w_lcq": _f(row, "w_lcq"),
        "salt_c_err": _f(row, "salt_c_err"),
        "w_time": _f(row, "w_time"),
        "w_mag": _f(row, "w_mag"),
        "w_prob": _f(row, "w_prob"),
        "w_salt": _f(row, "w_salt"),
        "w_iaspec": _f(row, "w_iaspec"),
        "n_points": _i(row, "n_points"),
        "salt_chi2_dof": _f(row, "salt_chi2_dof"),
        "salt_t0_err": _f(row, "salt_t0_err"),
        # detail-panel fields (2026-08-18): context for WHY a candidate ranks
        # where it does — phase anchors, template verdict, SALT shape/color,
        # host geometry, and the exposure divisor behind score_rate
        "peak_mjd": _f(row, "peak_mjd"),
        "exposure_minutes": _f(row, "exposure_minutes"),
        "template_best": _s(row, "template_best"),
        "template_margin": _f(row, "template_margin"),
        "rise_time": _f(row, "rise_time"),
        "nuclear_offset_arcsec": _f(row, "nuclear_offset_arcsec"),
        "salt_x1": _f(row, "salt_x1"),
        "salt_c": _f(row, "salt_c"),
        "host_morphology": _s(row, "host_morphology"),
        "surveys": _s(row, "surveys"),
        "brokers_detected": _s(row, "brokers_detected"),
        "offset_class": _s(row, "offset_class"),
        "needs_classification": _b(row, "needs_classification"),
    }


def build_summary(candidates: list[dict]) -> dict:
    surveys: dict[str, int] = {}
    for c in candidates:
        key = c.get("surveys") or "unknown"
        surveys[key] = surveys.get(key, 0) + 1
    spec = [c for c in candidates if c.get("tns_type")]
    zs = [c["z"] for c in candidates if c.get("z") is not None]
    nps = [c["n_points"] for c in candidates if c.get("n_points") is not None]
    return {
        "n_candidates": len(candidates),
        "surveys": surveys,
        "n_spec_classified": len(spec),
        "n_spec_ia": sum(1 for c in spec if c["tns_type"] == "SN Ia"),
        "median_z": round(median(zs), 4) if zs else None,
        "median_n_points": median(nps) if nps else None,
    }


# ---------------------------------------------------------------------------
# Light curves: ride in the payload (the deployed server has no nights/ dir).
# Fetched per candidate from Fink's ZTF portal, compacted to
# [[mjd, mag, magerr, band], ...] rows.
# ---------------------------------------------------------------------------
LC_MAX_POINTS = 150


def compact_lc(df, max_points: int = LC_MAX_POINTS) -> list | None:
    """DataFrame (mjd/magnitude/mag_err/band, FinkZTFClient.get_light_curve
    schema) -> compact [[mjd, mag, magerr, band], ...] rows, mjd-sorted,
    keeping the MOST RECENT ``max_points``. mjd rounded to 4dp, mags to 3dp;
    non-finite magerr -> null. None/empty input -> None (lc omitted)."""
    if df is None or len(df) == 0:
        return None
    points = []
    for _, r in df.iterrows():
        mjd, mag = r.get("mjd"), r.get("magnitude")
        if mjd is None or mag is None:
            continue
        mjd, mag = float(mjd), float(mag)
        if not (math.isfinite(mjd) and math.isfinite(mag)):
            continue
        err = r.get("mag_err")
        try:
            err = round(float(err), 3) if err is not None and math.isfinite(float(err)) else None
        except (TypeError, ValueError):
            err = None
        points.append([round(mjd, 4), round(mag, 3), err,
                       str(r.get("band", "?"))])
    if not points:
        return None
    points.sort(key=lambda p: p[0])
    return points[-max_points:]


def attach_light_curves(candidates: list[dict], fetcher=None) -> None:
    """Attach a compact ``lc`` field to each candidate with a ztf_oid.

    ``fetcher(ztf_oid) -> DataFrame`` defaults to Fink's ZTF portal
    (broker_clients.fink_ztf_client.FinkZTFClient.get_light_curve). A fetch
    failure for one object never kills the upload — lc is omitted with a
    warning. Candidates without a ztf_oid are skipped."""
    if fetcher is None:
        sys.path.insert(0, _REPO)
        from broker_clients.fink_ztf_client import FinkZTFClient
        client = FinkZTFClient()
        fetcher = client.get_light_curve
    with_oid = [c for c in candidates if c.get("ztf_oid")]
    logger.info("Fetching light curves for %d/%d candidates...",
                len(with_oid), len(candidates))
    n_ok = 0
    for i, cand in enumerate(with_oid, 1):
        try:
            lc = compact_lc(fetcher(cand["ztf_oid"]))
        except Exception as e:
            logger.warning("lc fetch failed for %s (%s); omitting",
                           cand["ztf_oid"], e)
            lc = None
        if lc:
            cand["lc"] = lc
            n_ok += 1
        if i % 20 == 0 or i == len(with_oid):
            logger.info("  light curves: %d/%d fetched (%d with data)",
                        i, len(with_oid), n_ok)


def load_night(night_dir: str) -> NightPayload:
    night_stamp = os.path.basename(os.path.normpath(night_dir))
    mjd = night_stamp_to_mjd(night_stamp)
    csv_path = os.path.join(night_dir, "candidates.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = [parse_candidate(r) for r in rows]
    # rank order = descending score_rate (the PI-approved primary ordering)
    # when the night carries it; old nights fall back to descending merit.
    sort_key = ("score_rate"
                if any(c["score_rate"] is not None for c in candidates)
                else "merit")
    candidates.sort(key=lambda c: (c[sort_key] is None,
                                   -(c[sort_key] if c[sort_key] is not None else 0.0)))
    if len(candidates) > MAX_CANDIDATES:
        logger.warning("%s: truncating %d candidates to %d",
                       night_stamp, len(candidates), MAX_CANDIDATES)
        candidates = candidates[:MAX_CANDIDATES]
    return NightPayload(night_stamp=night_stamp, mjd=mjd,
                        summary=build_summary(candidates),
                        candidates=candidates)


def upload_http(payload: NightPayload, api: str, key: str) -> dict:
    req = urllib.request.Request(
        api.rstrip("/") + "/v1/selection/nights",
        data=payload.to_json().encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def upload_db(payload: NightPayload, db_path: str) -> dict:
    sys.path.insert(0, _REPO)
    from api.selection import SelectionStore
    store = SelectionStore(db_path=db_path)
    return store.upsert_night(payload.night_stamp, payload.mjd,
                              payload.summary, payload.candidates)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("night_dir", help="e.g. nights/wide/ut20260818")
    ap.add_argument("--api", help="API base URL, e.g. http://localhost:8000")
    ap.add_argument("--key", help="bearer API key for --api mode")
    ap.add_argument("--db", help="write straight into this SQLite file (local dev)")
    ap.add_argument("--backfilled", action="store_true",
                    help="mark this night as a retrospective re-run (fits use "
                         "photometry/classifications as of upload time, not the night)")
    ap.add_argument("--no-lc", action="store_true",
                    help="skip light-curve fetching (fast re-upload path; "
                         "candidates carry no lc field)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.db and not (args.api and args.key):
        ap.error("give either --db PATH, or --api URL with --key KEY")

    payload = load_night(args.night_dir)
    if not args.no_lc:
        attach_light_curves(payload.candidates)
    if args.backfilled:
        payload.summary["backfilled"] = True
    if args.db:
        result = upload_db(payload, args.db)
    else:
        result = upload_http(payload, args.api, args.key)
    logger.info("%s: %s", payload.night_stamp, result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
