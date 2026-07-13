# CLAUDE.md — RubinAlerts

## Project Overview

Automated SN Ia candidate identification and spectroscopic follow-up system for Rubin LSST Deep Drilling Fields. Two subsystems:

1. **Alert Pipeline** (`run_tonight.py`): Aggregates alerts from 5 brokers (Fink, ANTARES, ALeRCE, ATLAS, TNS), fits light curves, computes merit scores, generates Magellan observing plans.
2. **LLAMAS Orchestrator** (`orchestrator/`): Spectroscopic scheduling for the MAGNETS collaboration — converts ranked candidates into LLAMAS observing plans with multi-program time accounting, composite prioritization, and standard star interleaving.

Full architecture doc: `docs/design/architecture.md`

## Environment

- **Conda env:** `RubinAlerts` (Python 3.12)
- **Activate:** `conda activate RubinAlerts`
- **Key dependencies:** astropy, pandas, numpy, sncosmo, antares-client, psycopg2-binary, requests, matplotlib, pyyaml

## Git Remotes

- `origin` — `stubbslab/RubinAlerts` (Chris's upstream, read-only for us)
- `fork` — `akumgill/RubinAlerts` (Akum's fork, push here)
- **Always push to `fork`**, not `origin`

## Running

```bash
# Alert pipeline — legacy DDF-centric mode
python run_tonight.py 61101 --min-prob 0.3 --days-back 30

# Alert pipeline — wide sky mode (payload-level selection: r<=21.5, z<=0.4,
# dec<=+22, fresh, variable/AGN-screened; Fink-only; ~3 min/night)
python run_tonight.py 61101 --sky-mode wide --output-dir nights/wide

# LLAMAS orchestrator — basic plan
python -m orchestrator plan --date 2026-10-15 --targets targets.csv --moon grey --output-dir output/

# LLAMAS orchestrator — full nightly with time accounting
python -m orchestrator run-nightly --date 2026-10-15 \
    --candidates candidates.csv --allocations ref/allocations_example.yaml \
    --moon grey --output-dir output/

# Post-night reconciliation
python -m orchestrator reconcile --allocations ref/allocations_example.yaml \
    --program MAGNETS-Stubbs --actual-hours 3.5 --moon grey --date 2026-10-15

# Backward compat (no subcommand = plan)
python -m orchestrator --date 2026-10-15 --targets ref/test_targets.csv --moon grey
```

## Code Conventions

- `@dataclass` for config and data models (not dicts)
- `logging.getLogger(__name__)` in every module
- `float('nan')` for optional numeric fields (check with `math.isfinite()`)
- Astropy `Time`, `SkyCoord`, `u.deg/u.hour` for all astronomical quantities
- Module-level default config instances (e.g., `LLAMAS_CONFIG = LLAMASConfig()`)
- Broker clients follow `BaseBrokerClient` interface with `query_alerts()`, `get_stamps()`

## Credentials

- RSP TAP: `~/.rsp_token`
- ATLAS: `~/.atlas_credentials` (INI format, `[atlas]` section)
- TNS: `~/.tns_credentials`
- ALeRCE DB: hardcoded in `broker_clients/alerce_db_client.py`

## Key Architecture Decisions

- **LLAMAS only** — orchestrator is for LLAMAS on Magellan/Baade exclusively. LDSS3 materials in `ref/` are reference only.
- **Multi-program time accounting** — allocations.yaml defines per-PI budgets across dark/grey/bright moon phases. Charge-on-schedule with post-night reconciliation.
- **Composite priority scoring** — science weight × budget factor × phase weight + observability + keyword signals. Phase weight (`w_time`) from alert pipeline boosts near-peak targets.
- **Greedy scheduling** — `score = composite_priority - airmass × 10`. Falls back to `(5 - priority) × 100 - airmass × 10` without prioritizer.
- **Exposure estimation cascade**: redshift table (proposal Table 1) → magnitude scaling (mag 20 = 45 min, 2.5x/mag) → fallback (45 min).
- **1-minute IFU overhead** (not 10 min like slit instruments).
- **Multiplicative merit** — alert pipeline merit = w_time × w_mag × w_prob × w_host × w_ext × w_broker. Must score well on ALL factors.

## Output Locations

- Alert pipeline: `nights/ut{YYYYMMDD}/`
- Orchestrator: `--output-dir` flag, files named `LLAMAS_{date}_{timeline,catalog,summary}.txt`
- Time accounting: `time_accounting.json` in output dir

## Knowledge Base

Central knowledge repository for this project and related research:
`~/Documents/knowledge-base/research/wiki/`

Relevant entries:
- `projects/rubinalerts.md` — Project overview, goals, timeline, MAGNETS context
- `methods/spectroscopic-scheduling.md` — Scheduling algorithm details, LDSS3 reference, LLAMAS specifics
- `methods/alert-broker-aggregation.md` — Broker client details
- `methods/light-curve-fitting.md` — Parabola + Villar fitting
- `methods/merit-scoring.md` — Merit function components

When making significant changes to RubinAlerts architecture or methods, update the corresponding knowledge-base entries to keep them in sync.

## Roadmap / Next Steps

(Items 1–6 of the old roadmap were resolved by the June 2026 design-review
loop — R17 Baade docstring, R3 state path, R8 quartile docs, etc. See
`docs/design/review-loop-log.md`.)

### HIGH PRIORITY (post wide-sky pivot, July 2026)
1. **ALeRCE-ZTF wide-sky reuse** — the DB client already queries the full ZTF
   footprint (~3k candidates) then discards 99.9% via the DDF filter; apply
   the wide-mode cuts instead. Second survey for the bright/nearby sample and
   the primary alert source during the August 2026 Rubin downtime.
2. **ANTARES proxy coalescing** — use its heuristic P(Ia) as a fallback only
   when no ML score exists (`effective_prob` + `prob_source` column,
   down-ranked, flagged `needs_classification`). Never average into ML probs.
3. **Ia-specific probability** — fold `f:clf_earlySNIa_score` / CATS class
   into w_prob (currently SN-vs-other only); scores are already fetched.

### MEDIUM PRIORITY
4. **Rubin RSP/TAP** — verify whether the token grants live Prompt Products;
   if so, direct DiaObject queries give an owned selection function and a
   Fink-outage fallback. Currently plumbed but never queried in the run path.
5. **SALT2-driven phase/typing** — replace parabola/Villar phase estimates
   with sncosmo SALT2 fits when a redshift exists (`--use-salt` exists but is
   effectively unused).
6. **Wide-mode multi-broker** — the aggregator drops payload selection
   columns (z_best, brightest_mag, xm_tns_*); carry them through so wide mode
   need not imply --fink-only. Also: per-object TNS xm should take the first
   non-null across alerts, not the most recent alert's value.

### PLANNED FEATURES
7. **Google Sheet integration** — Allow PIs to manually enqueue targets not sourced from alerts (gspread + service account). Design spec exists in `docs/design/spectroscopic-orchestration-review.tex` §"Planned: Google-Sheet manual-request front end".
