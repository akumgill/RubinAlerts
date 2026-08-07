#!/usr/bin/env python
"""Generate a queue-ready SN Ia candidate list from the live Fink-ZTF stream.

Built for the 2026-08 Chilean-storm downtime: Rubin's alert stream is dark and
the Chile-hosted ALeRCE broker is stale, but ZTF (Palomar, CA) via Fink
(France) is live. This pulls fresh, Magellan-observable, Ia-leaning SN
candidates for a given night and writes them in the MAGNETS queue CSV schema
so they can be seeded/submitted as a program's targets.

Usage:
    python scripts/generate_ztf_candidates.py 2026-08-13 \
        --program CfA-Stubbs --instrument LLAMAS --top 20 \
        --out ref/stubbs_llamas_2026-08-13.csv

Note: these are classifier-selected (Fink SNN/RF Ia score + TNS labels), NOT
yet run through the full light-curve-fit / merit / nuclear-AGN screen of
run_tonight.py. Treat as a vetted candidate pool for a human to prune.
"""
import argparse
import csv
import logging

from astropy.time import Time

from broker_clients.fink_ztf_client import FinkZTFClient

logging.basicConfig(level=logging.INFO, format="%(message)s")


def priority_for(ia_score: float) -> str:
    """Map a Fink Ia score to a queue priority tier."""
    if ia_score is None:
        return "P3"
    if ia_score >= 0.85:
        return "P1"
    if ia_score >= 0.70:
        return "P2"
    return "P3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="observing night, YYYY-MM-DD")
    ap.add_argument("--program", default="CfA-Stubbs")
    ap.add_argument("--instrument", default="LLAMAS")
    ap.add_argument("--days-back", type=float, default=15.0)
    ap.add_argument("--max-mag", type=float, default=21.5)
    ap.add_argument("--dec-max", type=float, default=22.0)
    ap.add_argument("--min-ia", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=20, help="keep the top-N by Ia score")
    ap.add_argument("--prefer-untyped", action="store_true",
                    help="for typing-driven programs: rank objects with no TNS "
                         "classification spectrum ahead of already-typed ones "
                         "(default off = decision-support only, Ia-score order)")
    ap.add_argument("--priority", default=None,
                    help="force ALL rows to this tier (e.g. P5) instead of the "
                         "Ia-score mapping — use for low/opportunistic fills "
                         "(e.g. nearby SNe during a survey gap)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mjd_now = int(Time(f"{args.date} 00:00:00").mjd)
    client = FinkZTFClient()
    if not client.available:
        raise SystemExit("Fink-ZTF API not reachable")

    df = client.fetch_fresh_sn_candidates(
        mjd_now=mjd_now, days_back=args.days_back, max_mag=args.max_mag,
        dec_max=args.dec_max, min_ia_score=args.min_ia)
    if df.empty:
        raise SystemExit("no candidates survived the cuts")

    if args.prefer_untyped and "tns_classified" in df.columns:
        # Untyped first (no classification spectrum yet), then by Ia score.
        df = df.sort_values(["tns_classified", "ia_score"],
                            ascending=[True, False], na_position="last")
    df = df.head(args.top)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["program", "name", "ra", "dec", "priority", "mag", "band",
                    "exposure_minutes", "instrument", "notes"])
        for _, r in df.iterrows():
            ia = None if r.get("ia_score") is None else float(r["ia_score"])
            tns = str(r.get("tns_type") or "").strip()
            tns_note = (f"TNS-classified {tns} (spectrum exists)" if tns
                        else "no TNS classification (untyped)")
            note = (f"ZTF/Fink live; class={r.get('fink_class', '')}; "
                    f"Ia_score={ia:.2f}; sn_score={float(r.get('sn_score', 0) or 0):.2f}; "
                    f"{tns_note}; last det MJD {float(r['mjd']):.1f}")
            tier = args.priority or priority_for(ia)
            w.writerow([args.program, r["objectId"], round(float(r["ra"]), 5),
                        round(float(r["dec"]), 5), tier,
                        round(float(r["magnitude"]), 2), r.get("band", ""),
                        "", args.instrument, note])
    print(f"\nwrote {len(df)} candidates -> {args.out}")


if __name__ == "__main__":
    main()
