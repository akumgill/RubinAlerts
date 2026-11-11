#!/usr/bin/env python
"""Build the Magellan CfA observing request package for a night.

The observer asks for the plan as the CfA LaTeX template compiled to PDF, and
as a FOLDER with the catalogue, the timeline and the finder charts beside it.
This assembles that folder from a night's plan:

    <PI>_Magellan_<MMDD>_<YYYY>/
        <INSTRUMENT>_<YYYYMMDD>_<PI>.pdf     the compiled template
        <INSTRUMENT>_<YYYYMMDD>_<PI>.tex     its source, for hand edits
        ObsPlan/<PI>_Magellan_<MMDD>_catalog     Magellan TCS catalogue
        ObsPlan/<PI>_Magellan_<MMDD>_timeline    the observer's night table
        Finders/                                 EMPTY — see below

Finder charts are NOT generated. The reference package carries one per target
and we have no chart generator; the folder is created with a README so the gap
is visible rather than silently missing from a submission.

Sources for the plan, in order of preference:
  --api/--key     live queue (GET /v1/plan/preview, or /v1/plan/block)
  --plan-json     a saved preview response, for working offline

Usage:
  python scripts/make_request_package.py --date 2026-11-02 \
      --api https://magnets-collab.onrender.com --key $KEY \
      --meta ref/request_package_stubbs.yaml --out-dir ~/Desktop
  # render only, skip pdflatex:
  ... --no-compile
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

logger = logging.getLogger(__name__)


def fetch_plan(api: str, key: str, date: str, instrument: str) -> dict:
    q = urllib.parse.urlencode({"date": date, "instrument": instrument})
    req = urllib.request.Request(f"{api.rstrip('/')}/v1/plan/preview?{q}",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def _is_standard(name: str) -> bool:
    """Standards are pseudo-targets named '<star>@am<lo>-<hi>', and the bright
    flux standards the collaboration actually uses are LTT/Feige/GD/G-numbers."""
    n = (name or "").upper()
    return ("@AM" in n or n.startswith("LTT") or n.startswith("FEIGE")
            or n.startswith("GD") or n.startswith("G191")
            or n.startswith("WDFS"))


def plan_to_night(plan: dict) -> dict:
    """Preview response -> the night dict request_package expects."""
    from orchestrator.obsfiles import split_subexposures
    targets = []
    for e in plan.get("timeline", []):
        n_exp, exp_sec = split_subexposures(e.get("exp_min"))
        targets.append({
            "name": e.get("target", ""),
            "priority": e.get("tier", ""),
            "mag": e.get("mag"),
            "n_exp": n_exp, "exp_sec": exp_sec,
            "is_standard": _is_standard(e.get("target", "")),
        })
    # The early-night calibration stack. Chris's real-time tool needs darks, a
    # twilight sky flat (fibre wavelength registration) and a continuum source
    # for flux normalisation, and the flat has to be taken IN TWILIGHT, so it
    # heads the timeline rather than sitting in the calibration section alone.
    pre = [
        {"what": "Twilight sky flat (evening twilight, before first target)",
         "detail": "Required for per-fibre wavelength registration; please "
                   "take before the first science target."},
        {"what": "Bright flux standard (continuum reference)",
         "detail": "Spectrum: 1x60s"},
    ]
    return {"date": plan.get("date"), "targets": targets, "pre_night": pre}


def plan_to_backup(plan: dict, queue: dict = None) -> list:
    """Overflow -> the Backup Program section.

    Two exclusions, both because offering the observer a bad option is worse
    than offering none:
      * targets tonight's sky rules out (the overflow carries the reason), and
      * STANDARDS — calibration is not a science backup, and a standard's
        airmass-binned pseudo-target is meaningless outside its slot.
    Exposures are looked up from the queue rather than defaulted, so a backup
    quotes the integration its submitter actually asked for.
    """
    from orchestrator.obsfiles import split_subexposures
    queue = queue or {}
    out = []
    for o in plan.get("overflow", []):
        name = o.get("target", "")
        reason = str(o.get("reason", "")).lower()
        if "airmass" in reason or "not observable" in reason:
            continue
        if _is_standard(name):
            continue
        q = queue.get(name, {})
        n_exp, exp_sec = (q.get("n_exposures"), q.get("exposure_seconds"))
        if not (n_exp and exp_sec):
            n_exp, exp_sec = split_subexposures(q.get("exposure_minutes"))
        out.append({"name": name, "priority": o.get("tier", ""),
                    "mag": q.get("mag"), "n_exp": int(n_exp),
                    "exp_sec": float(exp_sec), "is_standard": False})
    return out


def fetch_queue(api: str, key: str) -> dict:
    """name -> submitted row, for the exposures and magnitudes the plan
    preview does not carry through to the overflow."""
    req = urllib.request.Request(f"{api.rstrip('/')}/v1/targets",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req) as r:
            return {t.get("name"): t for t in json.loads(r.read()) if t.get("name")}
    except Exception as e:
        logger.warning("could not read the queue for backup exposures (%s)", e)
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True, help="UT date YYYY-MM-DD")
    ap.add_argument("--instrument", default="LLAMAS")
    ap.add_argument("--meta", default="ref/request_package_stubbs.yaml")
    ap.add_argument("--api")
    ap.add_argument("--key")
    ap.add_argument("--plan-json", help="saved preview response (offline)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--no-compile", action="store_true",
                    help="write the .tex but do not run pdflatex")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import yaml
    from orchestrator.obsfiles import request_package, tcs_catalog, night_plan

    meta = yaml.safe_load(open(args.meta))
    meta.setdefault("instrument", args.instrument)

    if args.plan_json:
        plan = json.load(open(args.plan_json))
    elif args.api and args.key:
        plan = fetch_plan(args.api, args.key, args.date, args.instrument)
    else:
        logger.error("give --plan-json, or --api and --key")
        return 2
    for w in plan.get("warnings", []):
        logger.warning("PLAN: %s", w)
    if not plan.get("timeline"):
        logger.error("no targets scheduled for %s — nothing to submit", args.date)
        return 1

    pi = meta["pi"]
    inst = meta["instrument"]
    mmdd = args.date[5:7] + args.date[8:10]
    yyyy = args.date[:4]
    stem = f"{pi}_Magellan_{mmdd}"
    pkg = os.path.join(os.path.expanduser(args.out_dir), f"{stem}_{yyyy}")
    obsplan = os.path.join(pkg, "ObsPlan")
    finders = os.path.join(pkg, "Finders")
    for d in (pkg, obsplan, finders):
        os.makedirs(d, exist_ok=True)

    # sibling files the PDF points at
    from orchestrator.obsfiles import split_subexposures
    rows = []
    for e in plan["timeline"]:
        n_exp, exp_sec = split_subexposures(e.get("exp_min"))
        rows.append({"name": e["target"], "ra": e["ra"], "dec": e["dec"],
                     "mag": e.get("mag"), "priority": e.get("tier"),
                     "n_exp": n_exp, "exp_sec": exp_sec,
                     "start": (e.get("utc") or "").split("-")[0],
                     "notes": ""})
    cat_name = f"{stem}_catalog"
    with open(os.path.join(obsplan, cat_name), "w") as f:
        f.write(tcs_catalog(rows))
    with open(os.path.join(obsplan, f"{stem}_timeline"), "w") as f:
        f.write(night_plan(rows))
    with open(os.path.join(finders, "README.txt"), "w") as f:
        f.write("Finder charts are NOT generated by the pipeline.\n"
                "One chart per target is expected here before submission.\n")

    meta["catalog_file"] = cat_name
    queue = fetch_queue(args.api, args.key) if (args.api and args.key) else {}
    meta["backup"] = plan_to_backup(plan, queue)
    tex = request_package([plan_to_night(plan)], meta)

    base = f"{inst}_{args.date.replace('-', '')}_{pi}"
    tex_path = os.path.join(pkg, base + ".tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    logger.info("wrote %s", tex_path)

    if args.no_compile:
        print(pkg)
        return 0
    if not shutil.which("pdflatex"):
        logger.error("pdflatex not on PATH — .tex written, compile it yourself")
        return 1
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         "-output-directory", pkg, tex_path],
        capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().split("\n")[-25:])
        logger.error("pdflatex failed:\n%s", tail)
        return 1
    for ext in (".aux", ".log", ".out"):
        try:
            os.remove(os.path.join(pkg, base + ext))
        except OSError:
            pass
    logger.info("wrote %s", os.path.join(pkg, base + ".pdf"))
    print(pkg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
