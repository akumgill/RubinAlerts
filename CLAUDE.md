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

### DONE (July 2026 sprint — see git log 8fd00e3..HEAD)
1. ~~**ALeRCE-ZTF wide-sky reuse**~~ — DONE: `query_fresh_sn_candidates`
   (time-filtered SQL, fixing the arbitrary-slice bug) feeds wide mode;
   class tags carried (`alerce_class`), Galactic-plane screen.
2. ~~**ANTARES proxy coalescing**~~ — DONE: `effective_prob` chain
   (mean_ia_prob → sn_score → proxy), `prob_source`, `needs_classification`.
3. ~~**Ia-specific probability**~~ — DONE: `w_iaspec` [0.8–1.2] from TNS
   spec-type / ALeRCE SNIa prob / earlySNIa; positive Ia evidence ≥ neutral.
   (CATS is broad taxonomy, carried but not Ia evidence.)
4. ~~**Rubin RSP/TAP**~~ — RESOLVED NO (audit 2026-07-13): token grants
   static DP1 only (ends 2024-12-12); no PPDB on community RSP; USDF is 401.
   Fink-independent Rubin selection impossible today; redundancy = brokers.
5. ~~**SALT2-driven phase/typing**~~ — DONE: survey-aware `fit_salt`
   (salt2-extended + F99 MW dust), `salt_z_policy` (spec-z fixed / photo-z
   bounded / free box), `choose_best_fit`, tiered rescue fits
   (`--salt-rescue-cap`). Default ON in wide mode, `--no-salt` to disable.
   Requires sncosmo + iminuit (iminuit absence was why --use-salt silently
   no-op'd historically).
6. ~~**Wide-mode multi-broker**~~ — DONE (aggregator phases 1+2):
   PASSTHROUGH_COLUMNS reducers carry payload columns through the merge;
   wide mode uses the real `merge_alerts` (2" tolerance) with cross-survey
   agreement stats; TNS xm first-non-null across alerts.

### NEXT UP
- ~~**Multi-type template tournament**~~ — DONE (2026-07-14, production):
  `fit_template` + `run_template_tournament` (core/peak_fitting) fit each
  finalist against SALT2 and nugent Ibc/IIP/IIn under the same z policy;
  `enrich_finalist_typing` (run_tonight Step 5b, `--no-tournament` to skip)
  writes template_best/template_best_chi2/template_margin/template_peak_mjd
  to candidates.csv. Ground truth 6/6 (4 Ia->Ia, TDE->IIn, SN II->IIP).
  REMAINING: feed verdict into merit/needs_classification; non-Ia rescue
  tier inside the fit loop (today's rescue is Ia-only); SLSN/TDE templates
  (wrapped community SEDs / magnetar) or ParSNIP as the all-types model.
- **Exotic top-of-funnel** — the selection funnel is Ia-shaped end-to-end
  (SN-classifier streams, Ia z/phase/freshness cuts, nuclear/long-baseline
  screens treat exotic signatures as contamination). The ranking/scheduling
  side is ready for a second program; the exotic group needs its own source
  streams + cuts (nuclear-allowed, baseline>150d allowed, hours-fresh,
  higher-z). Scenario runs 2026-07-13: exotic cohort = 12 past-peak leakage
  objects, 0 rising.
- ~~**Scheduler reconciliation (one authority per night)**~~ — DONE
  (2026-07-13): the pipeline RANKS, the orchestrator SCHEDULES.
  `run_tonight.py` now calls `orchestrator.run_nightly` at the end (Step 9;
  `--allocations`, `--moon-phase` auto-derived, `--no-orchestrate` to skip)
  and writes the executable plan to `nights/.../llamas/`. The pipeline's own
  observing_schedule.txt / magellan_plan.cat / optimized_sequence.csv are
  retired (generate_observing_schedule kept as deprecated dead code for one
  release cycle). NOTE: default allocations are the EXAMPLE file — replace
  with real MAGNETS budgets when agreed.
- ~~**TNS daily-dump cross-match**~~ — DONE (2026-07-13):
  `fetch_tns_public_objects` + `crossmatch_tns_local` (1 cached download,
  202k objects, internal-name-then-coordinate match); serial cone-search is
  the fallback. Needs bot_id/bot_name in ~/.tns_credentials.
- **ANTARES wide-sky query mode** — its client is a DDF-cone search; a
  dec-strip/all-sky query would let ANTARES join wide mode (~1 day).
- **Pipeline-side cross-night memory** — feed the orchestrator's target
  ledger back into nightly ranking (`observed_recently` flag / completeness
  in merit) so run_tonight doesn't re-rank an observed target #1; epoch
  cadence policy (per-program, on top of phase buckets) is a science
  decision for the collaboration.
- **Absolute cross-night merit thresholds** once statistics accumulate.
- **S/N-based exposure + w_mag replacement** (TODO(S/N-feasibility) marker).

### PLANNED FEATURES
7. **Google Sheet integration** — Allow PIs to manually enqueue targets not sourced from alerts (gspread + service account). Design spec exists in `docs/design/spectroscopic-orchestration-review.tex` §"Planned: Google-Sheet manual-request front end".
