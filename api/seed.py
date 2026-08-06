"""Demo seed for a fresh deployment.

When a freshly-provisioned instance has an empty queue and ``SEED_DEMO=1``,
this loads the **real Villar/Dong LDSS3 semester-26B target list** (from
``ref/seed_villar_ldss3.csv`` — checked into the repo, so it works on a Render
container with no access to the original Downloads CSV) so the dashboard shows
a real queue instead of a blank page. No fabricated stand-in targets: other
groups (UA, …) add their own via the API/UI once live.
"""
from __future__ import annotations

import csv
import logging
import math
import os

logger = logging.getLogger(__name__)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEED_CSV = os.path.join(_REPO, "ref", "seed_villar_ldss3.csv")
_ALLOCATIONS = os.path.join(_REPO, "ref", "allocations_LDSS3_2026B.yaml")


def load_seed_submissions() -> dict:
    """program -> list of submission dicts, parsed from the Villar seed CSV."""
    subs: dict = {}
    with open(_SEED_CSV, newline="") as f:
        for row in csv.DictReader(f):
            item = {
                "name": row["name"].strip(),
                "ra": float(row["ra"]), "dec": float(row["dec"]),
                "priority": row["priority"].strip(),
                "instrument": row.get("instrument", "LDSS3").strip() or "LDSS3",
                "notes": (row.get("notes") or "").strip(),
            }
            if row.get("mag") not in (None, ""):
                item["mag"] = float(row["mag"])
            if row.get("band"):
                item["band"] = row["band"].strip()
            if row.get("exposure_minutes") not in (None, ""):
                item["exposure_minutes"] = float(row["exposure_minutes"])
            subs.setdefault(row["program"].strip(), []).append(item)
    return subs


def ensure_demo_allocations(data_dir: str) -> str:
    """Path to the real LDSS3 allocations (the seeded night is LDSS3). ``data_dir``
    is accepted for signature compatibility but unused — we ship the file."""
    return _ALLOCATIONS


def demo_group_config() -> dict:
    """Default program -> {key, password} for a demo deploy: one login for the
    seeded Villar program plus UA (who add their own targets). Overridden by
    GROUPS_JSON / *_KEY / *_PASSWORD env in production."""
    return {
        "CfA-Villar": {"key": os.environ.get("CFA_KEY", "demo-cfa"),
                       "password": os.environ.get("CFA_PASSWORD", "cfa-demo")},
        "UA": {"key": os.environ.get("UA_KEY", "demo-ua"),
               "password": os.environ.get("UA_PASSWORD", "ua-demo")},
    }


def seed_demo(service) -> int:
    """Submit the real Villar seed into ``service``. Only programs with a key in
    the service are seeded (others skipped). Returns the count submitted."""
    prog_to_key = {prog: key for key, prog in service._programs.items()}
    total = 0
    for prog, subs in load_seed_submissions().items():
        key = prog_to_key.get(prog)
        if not key:
            logger.warning("seed: no API key for program %r; skipping %d targets",
                           prog, len(subs))
            continue
        results = service.submit(key, subs)
        ok = sum(1 for r in results if r.get("status") == "ok")
        total += ok
        bad = [r for r in results if r.get("status") != "ok"]
        if bad:
            logger.warning("seed: %d/%d %s targets failed: %s",
                           len(bad), len(subs), prog, bad[:2])
    logger.info("seeded %d demo targets", total)
    return total
