#!/usr/bin/env python
"""Stage a tabletop observing night from a nightly selection run.

Chris's exercise (2026-11-11): *"pretend like we have an observing night
coming up, and go through your broker arbitration thing and see where we end
up, and just make sure that we end up with a decent plan."*

The point is to rehearse the WHOLE path on a FUTURE night, because forward
prediction is the part most likely to be wrong and the part a real run always
needs. Dec 15 is over a month out; by then none of today's candidates are the
same objects. So what the exercise validates is the process, not the list —
which is exactly why the magnitude has to be faded to the target night rather
than taken at the epoch the pipeline happened to run.

FADING. candidates.csv carries peak_mag and delta_t (signed days from peak at
the run's own night). For a night ``n`` days later the phase shifts by n, and
the object dims by the coarse optical rates the orchestrator and the selection
page both use (LLAMASConfig: 0.10 mag/d rising, 0.05 mag/d declining, capped at
3 mag). Using peak_mag directly makes every exposure optimistically short.

Usage:
  python scripts/tabletop_night.py --night-dir nights/wide/ut20261111 \
      --date 2026-11-14 --api http://localhost:8000 --key <bearer> \
      --standards WDFS0122-30 --top 25
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import urllib.request
from datetime import date as _date

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

logger = logging.getLogger(__name__)


def fade(peak_mag: float, delta_t: float, config) -> float:
    """peak_mag + the coarse fade at phase ``delta_t`` (shared policy)."""
    from orchestrator.normalize import predicted_mag_at_observation
    return predicted_mag_at_observation(peak_mag, delta_t, config)


def build_items(night_dir: str, target_date: str, top: int, config) -> list:
    import pandas as pd
    df = pd.read_csv(os.path.join(night_dir, "candidates.csv"))
    run_night = os.path.basename(night_dir.rstrip("/")).replace("ut", "")
    run_d = _date(int(run_night[:4]), int(run_night[4:6]), int(run_night[6:8]))
    shift = (_date.fromisoformat(target_date) - run_d).days
    logger.info("fading %+d day(s) from the %s run to %s", shift,
                run_d.isoformat(), target_date)

    df = df.sort_values("score_rate", ascending=False)
    items, skipped = [], 0
    for i, (_, r) in enumerate(df.head(top).iterrows()):
        name = r["tns_name"] if isinstance(r.get("tns_name"), str) else r.get("ztf_oid")
        peak, dt = r.get("peak_mag"), r.get("delta_t")
        if not (isinstance(peak, (int, float)) and math.isfinite(peak)):
            skipped += 1
            continue
        dt_now = float(dt) if (isinstance(dt, (int, float))
                               and math.isfinite(dt)) else float("nan")
        dt_then = dt_now + shift if math.isfinite(dt_now) else float("nan")
        mag_now = fade(float(peak), dt_now, config)
        mag_then = fade(float(peak), dt_then, config)

        # Scale the PIPELINE'S OWN exposure by the brightness change rather
        # than re-deriving it from a different calculator: the pipeline sized
        # this target for its own night, and we only want the delta. In the
        # background-limited regime that applies to faint SNe on a 6.5 m,
        # S/N ~ F sqrt(t), so t ~ F^-2 = 10^(0.8 dm). (Source-limited would be
        # 10^(0.4 dm); the choice matters only for the brightest targets.)
        exp_now = r.get("exposure_minutes")
        if isinstance(exp_now, (int, float)) and math.isfinite(exp_now):
            dm = mag_then - mag_now
            exp_then = float(exp_now) * (10 ** (0.8 * dm)) if math.isfinite(dm) \
                else float(exp_now)
        else:
            exp_then = 30.0
        items.append({
            "name": str(name), "ra": float(r["ra"]), "dec": float(r["dec"]),
            "mag": round(float(mag_then), 2), "band": "r", "mag_kind": "point",
            "priority": "P1" if i < 6 else "P2", "instrument": "LLAMAS",
            "exposure_minutes": round(exp_then, 1),
            "notes": f"phase {dt_then:+.1f} d at {target_date}; "
                     f"mag {mag_now:.2f}->{mag_then:.2f}, exposure "
                     f"{float(exp_now):.0f}->{exp_then:.0f} min"
                     if math.isfinite(dt_then) else "phase unknown",
        })
    if skipped:
        logger.warning("%d candidate(s) had no peak_mag and were skipped", skipped)
    return items


def post(api: str, key: str, path: str, body) -> list:
    req = urllib.request.Request(
        api.rstrip("/") + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--night-dir", required=True, help="nights/wide/utYYYYMMDD")
    ap.add_argument("--date", required=True, help="the tabletop night, YYYY-MM-DD")
    ap.add_argument("--api", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--standards", default="",
                    help="comma-separated Boyd standards to ladder (optional)")
    ap.add_argument("--bins", default="1.0-1.3,1.7-2.3")
    ap.add_argument("--std-exposure-minutes", type=float, default=10.0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from orchestrator.config import LLAMAS_CONFIG
    items = build_items(args.night_dir, args.date, args.top, LLAMAS_CONFIG)
    res = post(args.api, args.key, "/v1/targets", items)
    ok = sum(1 for x in res if x.get("status") == "ok")
    for x in res:
        if x.get("status") != "ok":
            logger.warning("rejected: %s", x.get("error"))
    logger.info("science: %d/%d enqueued", ok, len(items))

    if args.standards:
        from subprocess import run
        cmd = [sys.executable, os.path.join(_REPO, "scripts", "enqueue_standards.py"),
               os.path.join(_REPO, "ref", "boyd2025_wdfs_standards.csv"),
               "--names", args.standards, "--bins", args.bins,
               "--exposure-minutes", str(args.std_exposure_minutes),
               "--api", args.api, "--key", args.key]
        run(cmd, check=False)
    print(json.dumps({"enqueued": ok, "date": args.date}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
