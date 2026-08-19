# RubinAlerts Lab Notebook

## 2026-08-19 — Stamped batch SHIPPED (items 1–8 all done; dry-run ready)

Eight commits (7abc565..07d0ccc), suite 389 → **415 passed** (+26). Pushed;
Render deployed. Everything from the 08-18 Chris 1:1 except the two
truly-external unknowns (LLAMAS macro dialect — Rob Simcoe; FITS delivery
location) is now live:

- **#3 G downweight**: typed-but-no-z 0.7 → 0.15 (host z via MOS later).
- **#4 East-rising factor E**: score = P×V×G×U×E; HA at evening twilight,
  east=1.0, meridian 0.9, half-cosine to 0.3 floor by HA+6h; feeds the
  score-chain display. Constants provisional in ScoreConfig.
- **#2 ETC pre-fill**: GET /v1/etc?mag= → 3×N triplet suggestion; add-target
  form fills only-if-empty, editable.
- **#1 One-click enqueue** from the selection detail panel: tier from rank
  (top5→P1, 6-15→P2, rest→P3), mag faded to tonight, ETC triplet, explicit
  ra/dec (no resolver dependency), dedupe-safe.
- **#5 Airmass ranges**: airmass_min/max end-to-end (hard window in planner;
  min-only + max-only both work; overrides max_airmass);
  scripts/enqueue_standards.py → per-bin pseudo-targets (GD71@am1.0-1.3),
  default bins 1.0-1.3/1.3-1.7/1.7-2.3.
- **#8 Standards ingested**: ref/boyd2025_wdfs_standards.csv — Boyd et al.
  2025 (MNRAS 540,385), 35 WDs from the Zenodo machine-readable table,
  Gaia-DR3-verified (faint 32 match <1"; the 3 CALSPEC primaries were
  1.6-3.6" off = older-epoch coords on high-PM WDs → re-anchored to Gaia,
  pm noted). Gaia G 11.7-20.0; 27/35 LCO-reachable. NOT enqueued to prod —
  pick 1-2 RA-appropriate per night (Chris), one command when wanted.
- **#6 Observation ingestion (mocked source)**: POST /v1/observations,
  pointing association ≤1' (standards disambiguated by airmass-at-obs-time),
  even split across programs sharing coords, idempotent by filename;
  burndown folds observed time into 'used'; "Observed" dashboard section +
  queue/selection observed-badges. FITS adapter isolated in
  scripts/ingest_fits_night.py (the ONLY swap point);
  scripts/mock_observations.py generates synthetic nights. Mocks seeded
  LOCAL DB only — deliberately NOT production (fake observed-markers on real
  targets would mislead the dry run).
- **#7 Observing-plan bundle**: POST /v1/obsplan (1-6 targets, triplets) +
  queue checkboxes → TCS catalog (column-exact vs real
  ref/march_obs_run/catalog.cat), LDSS3-convention plan sheet, LLAMAS macro
  PROVISIONAL pending Simcoe.

### SESSION END STATE (2026-08-19) — handoff for next session

**Everything committed & pushed** (fork/main = 07d0ccc + notebook commits);
Render deployed and healthy; working tree clean; no background jobs.

**Site data (magnets-collab.onrender.com):** 7 nights live under the FINAL
scoring (ut20260813–18 backfilled, ut20260819 = live night; 59–74 candidates
each; 69 persistent objects). Result of the meeting's changes: **AT 2026ydy
(the pre-peak object the old merit under-ranked) is now #1** — E=1.0 (eastern,
2 months of LC ahead), G=1.0 (untyped); AT 2026xle (+10d, ballooning exposure
25→36 min over the week) slipped to #2. Good demo narrative.

**Dry run: possible TODAY end-to-end** with two labeled stand-ins:
(a) LLAMAS macro file provisional (TCS catalog + plan sheet are REAL formats);
(b) mock FITS for the post-night half (scripts/mock_observations.py). Mocks
are LOCAL-ONLY by design — never ingest mocks into prod (fake observed-marks
on real targets would mislead).

**Open external asks (each = one-function swap when answered):**
1. LLAMAS macro dialect — ask Rob Simcoe / LCO docs → orchestrator/obsfiles.py
   llamas_macro().
2. Where FITS files land → scripts/ingest_fits_night.py header mapping.
3. Service-observer chat (triplet conventions; he's "the #1 user").

**Operational facts:**
- Credentials: ~/.magnets_credentials (GROUPS_JSON copy; Stubbs key inside).
- Nightly cycle (manual; cron parked): `python run_tonight.py <MJD> --sky-mode
  wide --output-dir nights/wide` then `python scripts/upload_selection_night.py
  nights/wide/ut<date> --api https://magnets-collab.onrender.com --key <key>`
  (--backfilled for retrospectives; MJD 61271 = 2026-08-19).
- Standards enqueue (pick 1-2 RA-appropriate per night, NOT all):
  `python scripts/enqueue_standards.py ref/boyd2025_wdfs_standards.csv
  --names GD71,... --api ... --key ...`
- Rubin stream still dark (newest Fink-LSST alert 2026-07-14); all sources ZTF.

**Backlog (non-blocking):** nightly cron automation; ScoreConfig constants
review with Chris (V(z) prior, G values, E shape, target binned S/N — the
last scales EVERY exposure); w_prob youth-tempering (pending program-intent
answer); optional continuous %-remaining budget factor; knowledge-base wiki
merit-scoring entry still describes old merit; LDSS3 long-slit deprioritized
for our targets (Villar-only).

## 2026-08-18 (later) — Chris 1:1 outcomes (dry run tomorrow) + enqueue-bug fix

### Meeting decisions (Flow notes; "YAMAS" = LLAMAS)

- **Program clarity:** spectrum = TYPE (perishable, the purity product); host
  redshifts DEFERRED to a later MOS campaign → downweight the z-component of
  the G info-gain ladder (typed-but-no-z drops ~0.7 → ~0.2; G ≈ "is the type
  unknown?"). Kills the salt_z-into-V(z) idea (precise z isn't the product).
- **East-rising preference:** prefer targets rising in the east at night start
  so post-peak PHOTOMETRY stays gettable ~2 months (a setting target's LC gets
  truncated by the sun → spectrum unusable for cosmology). This is the
  "forecast final LC sampling" factor from the 08-18 audit, now PI-endorsed.
  NOTE: the naive opposite reading ("observe setting ones before they're
  gone") is wrong for this program — a spectrum without a finishable LC is
  a wasted slot.
- **Accounting:** post-hoc NIGHTLY (not real-time) from FITS headers (object,
  exposure), associated to targets by pointing within ~1 arcmin, not name.
  Ledger is already coord-keyed. BLOCKED on: where FITS files land.
- **Observing files:** instrument-specific; triplets (e.g. 3x30) canonical for
  CR rejection; operator works in batches of 4–6 targets, not full nights.
  BLOCKED on format — Akum to ask Rob Simcoe / read LCO docs.
- **P-tier ties broken by % TIME REMAINING, not raw hours** — replaces the
  absolute-hours budget-factor tiers (1.0/0.5/0.1). Not yet implemented.
- **Pool LLAMAS/LDSS3 time in practice; skip long-slit LDSS3 for our own
  targets** (Villar keeps LDSS3). Affects accounting design.
- Standards: spectrophotometric standards need repeats across airmass ranges —
  model as separate fake objects per range; add per-target airmass-range
  option (default: minimize). Chris sending the standards list + paper.

### Transcript corrections & additions (vs the Flow summary)

- **Setting/rising CAUTION:** Chris's guidance is prefer RISING/EASTERN targets
  ("weight it in the direction of objects that are rising", "Look to the
  east"). Akum's end-of-meeting recap said it BACKWARDS ("prioritize things
  that are setting") and Chris's "precisely so" didn't catch it — trust the
  earlier verbatim, not the recap.
- **%-remaining tie-break is ALREADY IMPLEMENTED** (get_budget_factor is
  fraction-of-phase-allocation tiers, not absolute hours). Optional refinement:
  continuous factor instead of 3 tiers.
- NEW from transcript: exposure should AUTO-POPULATE from anticipated mag
  (tonight, not peak) on add-target; selection page needs one-click ENQUEUE
  (the copy-paste flow is where the demo stumbled); duplicate target across
  groups → split time evenly (manual for now); talk to the service observer
  ("30 min = 3×10?") — he is "the number one user of this tool"; follow-up
  demo with Chris after this iteration.

### STAMPED TO-DO LIST (2026-08-18, priority order)

1. Selection-page enqueue button (name/coords/mag-tonight/ETC-triplet/tier
   prefill) — ~2-3h  [IMPLEMENTING]
2. ETC pre-fill on manual add-target (resolved mag → editable suggestion)
   — ~1h  [IMPLEMENTING]
3. G downweight: typed-but-no-z 0.7 → 0.15 — minutes  [IMPLEMENTING]
4. East-rising factor (hour angle at evening twilight) — ~1-2h  [IMPLEMENTING]
5. Per-target airmass range + standards as fake objects per airmass bin
   — ~half day  [IMPLEMENTING]
6. Observation ingestion + accounting + observed-repository — UNBLOCKED via
   mock: canonical record + POST /v1/observations + pointing association
   (1 arcmin; standards disambiguated by airmass-at-obs-time) + budget
   deduction (split evenly on cross-program dupes) + "Observed" dashboard
   section + observed-badges informing later ranking (cross-night memory).
   Only the FITS-header adapter waits on the real delivery location/dialect.
   ~1 day  [IMPLEMENTING as item F, after 1-5]
7. "Generate observing plan" button (batches 4-6, 3×N triplets) — UNBLOCKED:
   2 of 3 output layers have REAL in-repo formats (ref/march_obs_run/
   catalog.cat = genuine Magellan TCS catalog; ref/LDSS_ObsPlan_Generator/
   example_targets.txt = plan-sheet convention with "3x900s" triplets); only
   the LLAMAS instrument-macro dialect waits on Rob Simcoe, isolated behind a
   provisional serializer. ~1 day  [IMPLEMENTING as item G, after F]
8. Standards list ingestion — small — BLOCKED: paper/electronic table incoming
9. Optional: continuous %-remaining budget factor — trivial
10. Human: Rob Simcoe, service-observer chat, Villar-night trip, time card

### Enqueue bug FIXED (dry-run blocker)

Root cause: `api/app.py` built TargetQueueService with **resolver=None** —
EVERY name-only submission failed (not just long ZTF names); only tests had a
(fake) resolver. Second layer: Fink's TNS resolver matches only the exact full
name ("AT 2026ydy"; bare "2026ydy" returns []). New `api/resolver.py`
(public Fink API, no credentials): ZTF ids via /api/v1/objects (latest-alert
ra/dec+mag), TNS names via /api/v1/resolver with AT/SN-variant expansion.
Verified end-to-end: ZTF26abmiytv and 2026ydy both enqueue; coord-dedup
recognized them as the same target (the ~1-arcmin association working).
Suite 389 passed. Commit cb7a888.

---

## Project Overview

Automated SN Ia candidate identification pipeline for Rubin LSST Deep Drilling Fields. Aggregates alerts from multiple brokers (Fink, ANTARES, ALeRCE), fits light curves, computes spectroscopic follow-up merit scores, and generates Magellan observing plans.

---

## 2026-08-18 — Nightly run, Rubin-stream status, sampling-in-merit audit, Stubbs selection tab

### Nightly run (MJD 61270 → `nights/wide/ut20260818/`)

56 ranked candidates, all SALT2-fit (49 with chi2/dof < 2); 55/56 already on TNS,
29 spec-classified (26 Ia); median z = 0.07; 4 nuclear flags. LLAMAS plan +
accounting written (Stubbs 23.0h / Villar 16.0h remaining).

### Rubin alert stream is DARK — all sources are ZTF

All 56 finalists came via Fink-ZTF. Fink-LSST returned 3584 alerts but **0 pass
the 30d freshness cut**: newest alert across all three tags is **MJD 61235 ≈
2026-07-14** — the stream went quiet ~2 weeks *before* the "early–end Aug"
downtime window in `ref/observing_nights_2026B.yaml`. Can't disambiguate Fink
ingestion vs Rubin itself (no Fink-independent access per 2026-07-13 audit).
ASK CHRIS: expected return date; December (prime DDF season, his Dec 15 night)
is the run that really needs it.

### Light-curve sampling is NOT in the merit — and w_salt anti-selects for it

No merit factor measures sampling; only gate is >=5 pts SNR>5, >=2 bands.
Tonight's #1/#2/#3 have 6/5/8 points and all take max w_salt=1.2 (SALT2 through
~5 pts is near-interpolation, chi2/dof 0.14–0.33). Spearman(merit, n_points) =
**−0.25**. Matches the Aug 7 review P-item. Nuance: sparse-now ≠ sparse-forever
(young SNe accrete points), so don't naively reward n_points — it fights w_time.
Proposed: persist `x1_err`/`c_err` (computed in `fit_salt` peak_fitting.py:588-93
but dropped; only `salt_t0_err` saved), gate on t0_err ≲ 2d ("rise constrained"),
make the chi2 bonus conditional on points/dof.

### Meeting prep pointers

Exposure-time state: `docs/design/program-questions-for-chris.md` (open knobs:
target binned S/N, n_bin/R~200 sufficiency, max per-target exposure). DDF
rationale: distances come from the photometric SALT2 fit — spectrum is only
type+z, so spectra spent on poorly-sampled LCs are wasted for cosmology; ZTF
covers this adequately only at z ≲ 0.15–0.2 and can't see southern DDFs.
Dates (`ref/observing_nights_2026B.yaml`): next Sep 6/7 (LLAMAS halves, Yize);
Chris: Dec 15 + Jan 12.

### NEW RANKING (PI-approved): score = P × V(z) × G × U, ordered by score_rate

Replaces merit_rate as the primary ordering (merit columns retained). P =
w_prob × w_iaspec × w_lcq (new LC-quality from salt_c_err — x1/c errors now
persisted); V(z) = inverse sample density per Δz=0.05 bin (ledger + community-
saturation prior; supersedes orchestrator z_preference, now default-OFF); G =
info gain (0.05 when spec-typed AND spec-z'd → "free sample"); U = deadline-
shaped urgency in REST-frame days (fixes (1+z) bias; feeds orchestrator
phase_weight). ÷T_exp via the July α=0.5 density decision. All constants in
`ScoreConfig` (config.py), provisional pending Chris. Decision log:
`docs/design/review-loop-log.md` 2026-08-18. Suite: 375 passed.

**Effect (Aug 17 night):** 31/60 candidates are free-sample — half the ZTF-era
queue was re-confirming community knowledge; top-8 now all unclassified ATs.
Score~n_points Spearman −0.14 (merit was −0.19); sparse young objects pay the
w_lcq 0.4 floor instead of collecting the w_salt bonus. v_z uniformly 0.08–0.15
(low-z saturated per prior) — quantifies "the value starts when Rubin returns."

### Backfill: 6 consistent nights (Aug 13–18) in the selection tab

Reran MJD 61265–61270 under the new code (61265–69 flagged `backfilled` — fits
use photometry/classifications as of TODAY, not the night; labeled in UI).
65 persistent objects; stable top cohort all week (AT 2026xle rank 1 all six
nights; AT 2026xcr 4–5; AT 2026wyf exactly 7 nightly). Caveat: shared
present-day photometry makes backfilled persistence read more stable than live.

### NEW: Stubbs-only "SN Ia target selection" web tab (uncommitted)

Per-night ranked-candidate view + cross-night persistence (rank trails), gated
to `SELECTION_PROGRAMS` (default CfA-Stubbs) at the API; nightly results
uploaded via `scripts/upload_selection_night.py` (Render never sees `nights/`).
Files: `api/selection.py`, endpoints in `api/app.py`, `web/selection.{html,js}`,
tests in `tests/orchestrator/test_selection_api.py`. Suite: 361 passed, 4
skipped. Seeded ut20260713/0813/0818 locally. Deploy: no migration; optionally
set `SELECTION_PROGRAMS`; upload nightly with a real bearer key from GROUPS_JSON.
Gotcha found: CSV `merit_rank` orders by merit_rate (merit/exposure), not merit.

---

## 2026-04-18 — Phases 2-4: Time Accounting, Prioritizer, Integration

### Phase 2: Time Accounting & Prioritizer

Built multi-program time accounting for the MAGNETS queue:

- **`accounting.py`** — `TimeAccountant` class: loads allocations from YAML, charges time per program per moon phase (D/G/B), supports post-night reconciliation. Persists state to `time_accounting.json` with full charge log for audit.
- **`prioritizer.py`** — Composite scoring: `science_weight × budget_factor × phase_weight + observability + keyword_adj`. Phase weight uses `w_time` from alert pipeline (Gaussian decay from peak) — near-peak targets automatically prioritized.
- **`allocations_example.yaml`** — Stubbs 30h (5D+20G+5B), Villar 16h (4D+8G+4B).

Budget factor tiers: 1.0 if >5h remaining, 0.5 if 0-5h, 0.1 if exhausted.

### Phase 3: RubinAlerts Integration

- **`run_nightly.py`** — End-to-end: `candidates.csv` → estimate exposures → calculate twilight → rank targets → schedule → charge time → write outputs.
- **CLI subcommands**: `plan` (original), `run-nightly` (with time accounting), `reconcile` (post-night adjustment). Backward compatible — no subcommand falls through to `plan`.
- `normalize.py` now propagates `default_program` and `phase_weight` (w_time) from candidates.csv.

### Phase 4: Reporting

- **`reporting.py`** — Nightly time report (per-program charges, budget status) and season progress report (cumulative usage, burn rate, projected exhaustion).
- `write_summary()` now includes Time Budget section when accountant provided.

### Validation

Tested full pipeline on Yize's targets:
- `run-nightly`: 2 targets scheduled, 4.7h charged to MAGNETS-Stubbs grey budget
- `reconcile`: Weather lost 1.2h → returned to budget (26.5h remaining)
- Backward compat: `--date --targets` without subcommand still works
- Season report shows burn rate (3.5h/night) and projected nights remaining

### Architecture Document

Created `docs/design/architecture.md` — comprehensive system architecture covering both the alert pipeline and LLAMAS orchestrator, suitable for design review.

### Files Added/Changed

```
orchestrator/accounting.py       — NEW: Multi-program time accounting
orchestrator/prioritizer.py      — NEW: Composite priority scoring
orchestrator/run_nightly.py      — NEW: End-to-end nightly orchestration
orchestrator/reporting.py        — NEW: Time and season reports
ref/allocations_example.yaml     — NEW: Example MAGNETS allocations
docs/design/architecture.md      — NEW: Full system architecture doc
orchestrator/models.py           — CHANGED: program, phase_weight, ProgramAllocation
orchestrator/planner.py          — CHANGED: prioritizer_scores + accountant integration
orchestrator/cli.py              — CHANGED: Subcommands (plan, run-nightly, reconcile)
orchestrator/normalize.py        — CHANGED: default_program, w_time propagation
orchestrator/output.py           — CHANGED: Budget section in summary
CLAUDE.md                        — CHANGED: Updated with full command reference
```

---

## 2026-04-17 — Orchestrator Phase 1 Complete & Broker Audit

### Phase 1 Implementation

Built the full LLAMAS orchestrator package (`orchestrator/`) with three parallel agents (architect, engineer, reviewer). Modules:

| Module | Purpose |
|--------|---------|
| `models.py` | Target, ScheduledEntry, ObsPlan dataclasses |
| `normalize.py` | CSV/RubinAlerts loader, exposure estimation (redshift table → mag scaling → fallback) |
| `planner.py` | Twilight, airmass, observability, greedy scheduler, WD standard selection |
| `output.py` | Timeline, Magellan TCS catalog, human-readable summary |
| `config.py` | LLAMASConfig (LCO site, 1-min IFU overhead, exposure table from proposal) |
| `cli.py` | `python -m orchestrator --date --targets --moon --output-dir` |

**Scheduling algorithm:** Greedy scorer: `score = (5 - priority) * 100 - airmass * 10`. Adapted from Alex's LDSS3 script but with LLAMAS-specific overhead (1 min vs 10 min slit setup).

**Critical bugs caught by reviewer:**
- Standards parser broke on multi-word names like "NGC 7293" — fixed with regex delimiter
- Post-processing gap-fill cascade could push entries past morning twilight — simplified to extend-last-only
- `n_exp` formula was mathematically wrong at edge cases — replaced with `ceil(total_sec / 900)`

### Orchestrator Validation

**DDF test (8 targets):** 7 scheduled at 99% efficiency, 1 backup. Working correctly.

**Yize's targets test (12 from `ref/test_targets.csv`):** 2/12 observable on Oct 15 from LCO (most are northern, not visible). The 2 southern targets scheduled correctly at 55% efficiency with proper standard star bookends. Low yield expected — targets were for a March run.

### Broker Status Audit

Tested all 5 broker clients live:

| Broker | Status | Fix Applied |
|--------|--------|-------------|
| Fink LSST | Working | — |
| ALeRCE (ZTF+LSST) | Working | — |
| ANTARES | Working | Was missing conda env activation (package installed in RubinAlerts env) |
| ATLAS | Working | Added retry/backoff to auth + token expiry re-auth on 401/403 |
| TNS | Working | Credentials fine, was skipped in `--fink-only` mode |

**ATLAS changes (`broker_clients/atlas_client.py`):**
- `_ensure_token()` now retries 3× with exponential backoff (5s, 10s, 20s)
- New `_request_with_reauth()` transparently re-authenticates on expired tokens
- All request methods use it (submit, poll, download, cleanup, batch)
- Added 30s/60s timeouts to prevent hanging

### Files Added/Changed

```
orchestrator/                    — NEW: Full LLAMAS scheduling package (7 files)
ref/test_targets.csv             — NEW: 12 targets from Yize's Google Sheet
docs/design/orchestration-summary.tex/.pdf — NEW: 1-page summary for advisor
broker_clients/atlas_client.py   — CHANGED: retry/backoff, token re-auth
LAB_NOTEBOOK.md                  — CHANGED: this entry
```

---

## 2026-04-14 — Spectroscopic Orchestration Design

### Meeting with Yize Dong (Ashley's group)

Discussed their current spectroscopic follow-up workflow:

**Current process:**
1. **Target requests**: Google Form → Google Sheet (Requester, Instrument, Target, RA/Dec, Brightness, YSE-PZ link, etc.)
2. **Obs plan generation**: Yize manually runs a notebook, ad-hoc ranking by visibility + brightness
3. **Output**: Raw data → Google Drive → processed → SkyPortal

**Existing tooling:**
- Alex has an LDSS3 script (`generate_obsplan.py`) that does greedy scheduling with priority support
- Yize doesn't currently use it due to input format mismatch
- Script is good reference for scheduling algorithm

**Key quote from Chris:** "Needs to be much more automated including some kind of time accounting."

### MAGNETS Collaboration Context

Read the Stubbs 2026B proposal (propid 2835). Key points:

**MAGNETS** = Magellan partner institutions pooling time for Rubin transient follow-up:
> "Our collaboration's plan is to pool our awarded observing time and develop an internal queue schedule to address a wide range of science goals... We will apportion the queue observing time and target prioritization in proportion to the time awarded by the respective TACs."

**My role (from proposal addendum):**
> "Graduate students Akum Gil and Jonah Medoff... will contribute to the collaborative infrastructure needed to execute this blended program and to distribute the data."

**Stubbs 2026B allocation:**
- Instrument: **LLAMAS** (integral field spectrograph on Magellan/Baade)
- 3 nights: 0.5D + 2.0G + 0.5B = 30 hours total
- Targets: SNe Ia in DDFs, z=0.1-0.4, r=18-21.5 mag
- Queued observing: Yes
- Science goal: Test DESI evolving dark energy hypothesis

**LLAMAS advantages over slit spectroscopy:**
- ~1 min overhead (vs ~10 min for slit setup)
- No slit losses (IFU)
- Host galaxy spectrum "for free" in FOV

### Design Doc Created

Wrote `docs/design/spectroscopic-orchestration.md` covering:

| Component | Purpose |
|-----------|---------|
| Ingester | Pull from RubinAlerts candidates + manual requests |
| Normalizer | Convert to common schema, estimate exposures by redshift |
| Prioritizer | Merit × observability × moon phase compatibility |
| Time Accountant | Track 30-hour allocation across moon phases |
| LLAMAS Planner | Greedy scheduling with WD standard interleaving |

**Moon phase scheduling** (from proposal Table 1):

| Redshift | Moon | Peak r | Exposure |
|----------|------|--------|----------|
| 0.10-0.20 | Bright | 19.3 | 35 min |
| 0.20-0.30 | Grey | 20.6 | 95 min |
| 0.30-0.35 | Grey/Dark | 21.3 | 45-160 min |
| 0.35-0.40 | Dark | 21.9 | 180 min |

**Note:** LDSS3 materials are reference only — this program uses LLAMAS exclusively.

### Files Added/Updated

```
docs/design/spectroscopic-orchestration.md  — NEW: Full design doc
ref/LDSS_ObsPlan_Generator/                 — Reference LDSS3 scheduling script
ref/march_obs_run/                          — Example obs plan from Yize
ref/cstubbs2026B (2) (1).pdf               — Stubbs 2026B proposal
memory/project_ashley_group_workflow.md     — MAGNETS context
```

### Next Steps

- [ ] Phase 1: Normalizer module, LLAMAS Planner, basic CLI
- [ ] Phase 2: Time accounting, prioritizer
- [ ] Phase 3: RubinAlerts integration
- [ ] Phase 4: Reporting, WD standards

---

## 2026-03-27 - Stubbs group meeting

- Still in a holding pattern with Magellan LLAMAS followups work, but I should read more about that and get familiar with the instrument and the integral field unit (?) spectrograph
- Chris says Rubin has started generating alerts? I should check the pipeline again and see if we're picking them up. If not, it might be some auth thing? IDK

## 2026-03-25 - Meeting w/ Chris Notes

Motivation:

- LLAMAS in Magellan. 1/2 arc minute range so much better obs efficiency
- Pooling magellan time/data so we get high cadence on Rubin things
- Most folks interested in TypeII SNe
- SNe Ia -
  - spectra gives us the redshift. Can get redshift after the fact as well
  - diff classes of SNe differentiate in spectra more than photo
  - host galaxy in FOV- get the galaxy spectra for free. Can correlate residuals against host galaxy properties (incl extinction). We know which are ellipticals

Implementation:

- proposal deadline in a few weeks
- toolkit pieces we need are:

  - (working) some way for us to give a rank ordered list of preferred candidates
  - merge and rank requests from all participants and turn that into an observing plan
    - diff PIs met in the past few days
    - [done] message the group saying that I can do the coordination
    - [todo] Chris/Akum to chat with Ashley about what to do
  - Stubbs/Villar to meet up and do a joint proposal. TBD on time allocation. Each institution will bring its list and time allocated to institutional night. Weights in proportion to time contributed.
  - Carve out and protect individual science projects for ind. grad students
    - Science goal for me: effect of galaxy morphology on SNe Ia

- Next steps: Chris kicks off slack channel

## 2026-03-15 — Rise Time Constraints Filter

### Session Summary

Added rise time filter to reject slow-rising transients that are unlikely to be SNe Ia.

### Implementation

**Science basis:** SNe Ia rise from explosion to peak in ~17-20 days. Slower transients (Type IIP SNe, some TDEs) take longer.

**Rise time calculation:**

- For Villar fits: `rise_time = peak_mjd - shared_t0` (explosion epoch from fit)
- For parabola fits: `rise_time = peak_mjd - first_detection_mjd` (approximate)

**Filter criteria:**

- Reject if `rise_time > 30 days` (configurable via `--max-rise-time`)
- Reject if `rise_time < 5 days` (unphysical, bad fit)

**Output:** `rise_time` column added to `candidates.csv`

**CLI:**

```bash
python run_tonight.py 61100 --max-rise-time 25  # stricter filter
```

**Files changed:**

```
run_tonight.py  — Added rise time computation, filter, CLI argument, CSV output
```

---

## 2026-03-13 — ZTF Batch Photometry Optimization

### Session Summary

Optimized ZTF photometry fetching from per-candidate REST API calls to batch database queries.

### Changes

**Problem:** Per-candidate ZTF queries via ALeRCE REST API were slow (~1-2 sec each), creating significant overhead for large candidate lists.

**Solution:** Added batch fetching via direct PostgreSQL access:

1. `AlerceDBClient.crossmatch_positions()` — spatial cross-match of candidate positions to ZTF OIDs using box query with spherical distance filter
2. `fetch_ztf_photometry_batch()` — batch fetch all ZTF detections in single DB query, convert to nJy flux format

**Performance:** 3 candidates × 85 detections fetched in ~1 second (vs ~6 seconds with REST API).

**Files changed:**

```
broker_clients/alerce_db_client.py  — Added crossmatch_positions() method
run_tonight.py                      — Added fetch_ztf_photometry_batch(), updated Pass 2 to use batch
```

**Note:** Most Rubin transients won't have ZTF counterparts (fainter, newer, or in southern DDFs). The batch optimization helps when there ARE matches.

---

## 2026-03-13 — False Positive Rejection Pipeline

### Session Summary

Added three major features for reducing false positives and avoiding duplicate discoveries: ATLAS credential verification, TNS cross-matching, and nuclear offset filtering.

### 1. ATLAS Forced Photometry (Credential Verification)

**Status:** Integration complete, awaiting valid credentials.

The ATLAS forced photometry integration was already implemented, but silently failing due to credential issues. Added:

- `verify_credentials()` method to AtlasClient
- Startup verification before pipeline runs
- Clear error messages for troubleshooting

ATLAS provides ~2 years of pre-discovery baseline photometry in cyan (c) and orange (o) filters.

### 2. TNS Cross-Match

**New file:** `broker_clients/tns_client.py`

Cross-matches candidates against IAU Transient Name Server to:

- **Avoid duplicating known discoveries** — objects already reported to TNS
- **Validate classifications** — compare our P(Ia) with spectroscopic confirmations
- **Get redshifts** — additional source beyond NED for SALT fitting

**Columns added:** `tns_name`, `tns_type`, `tns_redshift`, `tns_match`

**Usage:**

```bash
python run_tonight.py 61100           # TNS enabled by default
python run_tonight.py 61100 --no-tns  # Skip TNS
```

### 3. Nuclear Offset Filter

**Purpose:** Distinguish SNe from nuclear transients (AGN, TDE)

Uses host galaxy positions from GLADE+, SDSS, PS1, SkyMapper to compute the angular separation between transient and host nucleus.

**Classifications:**
| Offset | Classification | Interpretation |
|--------|---------------|----------------|
| < 1" | `nuclear` | Likely AGN or TDE (flagged with warning) |
| 1-30" | `offset` | Consistent with SN in host galaxy |
| > 30" | `distant` | May not be associated with detected host |

**Columns added:** `nuclear_offset_arcsec`, `offset_class`

**Example output:**

```
Nuclear offset: 2 NUCLEAR (likely AGN/TDE), 45 offset (SN-like), 3 distant
  *** 2 candidates are NUCLEAR (potential AGN/TDE) ***
    Nuclear candidates: obj_12345, obj_67890
```

### Files Changed

```
broker_clients/atlas_client.py   — Added verify_credentials()
broker_clients/tns_client.py     — NEW: TNS cross-match client
broker_clients/__init__.py       — Export TNSClient
host_galaxy/morphology_filter.py — Nuclear offset calculation
run_tonight.py                   — TNS + offset integration, --no-tns flag
```

---

## 2026-03-13 — Data Population Analysis & Fixes

### Session Summary

Diagnosed why merit function weights were defaulting to 1.0. Found critical bugs in E(B-V) propagation and host morphology classification. Added GLADE+ galaxy catalog.

### Data Population Audit

Analyzed 64 final candidates to identify where default values were being used:

| Parameter      | Default Rate | Root Cause                     | Fix Applied |
| -------------- | ------------ | ------------------------------ | ----------- |
| `w_ext` = 1.0  | 100%         | E_BV column never computed     | ✓ Fixed     |
| `w_host` = 0.7 | 98%          | Catalog queries returning NULL | ✓ Fixed     |
| `n_ztf` = 0    | 100%         | ALeRCE photometry not fetched  | Planned     |
| `n_atlas` = 0  | 100%         | ATLAS forced phot disabled     | Planned     |

### Bug Fixes

#### 1. Extinction E(B-V) Not Propagating

**Problem:** `get_extinction_batch()` returned `A_u`, `A_g`, `A_r`, `A_i`, `A_z` columns, but `run_tonight.py` looked for `E_BV` which didn't exist.

**Diagnosis:** Cache showed 506/507 positions had `extinction_json` stored, but `w_ext` was 1.0 for all candidates.

**Fix:** Compute E(B-V) from A_g using Schlafly & Finkbeiner (2011) coefficients:

```python
E_BV = A_g / R_g   # R_g = 3.303
```

#### 2. Host Morphology 98% Unknown

**Problem:** SDSS, Pan-STARRS, SkyMapper queries returning NULL for faint DDF galaxies.

**Diagnosis:** Cache showed 506/507 records with `morphology = NULL` and `catalog = NULL`. Only 1 galaxy found (SDSS elliptical at z~0.04).

**Fixes:**

- Increased search radius: 1 → 2 arcmin (SNe can be offset from host centers)
- Added Pan-STARRS fallback for all dec > -30 (deeper than SDSS)
- Added GLADE+ galaxy catalog as final fallback (22M galaxies with redshifts)
- Return 'uncertain' instead of 'unknown' when galaxy found but no optical colors

### New Catalog: GLADE+

Added `CatalogQuery.query_glade()` for GLADE+ (Galaxy List for Advanced Detector Era):

- VizieR catalog VII/291
- 22 million galaxies optimized for GW follow-up
- Provides spectroscopic/photometric redshifts
- All-sky coverage, good for southern DDFs

**Limitation:** Most GLADE+ entries lack B-band photometry (only WISE W1/W2), so morphology classification returns 'uncertain'. But we still get:

- Confirmation of host galaxy presence
- Redshift for distance modulus calculation
- Galaxy position for offset measurements

### Files Changed

```
utils/extinction.py              — Added E_BV computation, SFD_R_COEFFICIENTS
utils/catalog_query.py           — Added query_glade(), improved classify_morphology()
host_galaxy/morphology_filter.py — 2 arcmin search, PS1 fallback, GLADE+ fallback
core/magellan_planning.py        — Added w_salt, w_absmag to merit function
run_tonight.py                   — SALT fitting hooks, NED redshift support
```

---

## 2026-03-13 — Major Pipeline Enhancements

### Session Summary

Comprehensive code review and refactoring session. Added merit breakdown visualization, optimized observing sequence scheduler, Fink enrichment for ANTARES candidates, and centralized configuration.

### New Features Implemented

#### 1. Merit Function Breakdown

**Problem:** Reports showed only final merit score with no insight into what was driving rankings.

**Solution:** Added `compute_merit_breakdown()` function and report visualization.

- Individual weight columns: $W_{\rm time}$, $W_{\rm mag}$, $W_{\rm prob}$, $W_{\rm host}$, $W_{\rm ext}$, $W_{\rm broker}$
- Merit breakdown table in PDF reports
- Reference page explaining each parameter

**Merit Function:**

```
Merit = W_time × W_mag × W_prob × W_host × W_ext × W_broker

W_time   = exp(-Δt²/2τ²)           τ = 10 days, Gaussian decay from peak
W_mag    = Gaussian(m_opt=20.5)    Optimal for Magellan spectroscopy
W_prob   = P(Ia) ∈ [0.1, 1.0]      ML classifier probability
W_host   = {1.0, 0.6, 0.7}         Elliptical, Spiral, Unknown
W_ext    = exp(-E(B-V)/0.15)       Galactic extinction penalty
W_broker = 1.0 + 0.1×(N-1)         Multi-broker agreement bonus
```

#### 2. Fink Enrichment for ANTARES Candidates

**Problem:** ANTARES-only candidates had proxy P(Ia) = 0.30 (heuristic) instead of real ML scores.

**Solution:** Added `FinkLSSTClient.get_classifications()` for coordinate cross-match.

- Cross-matches ANTARES candidates against Fink by position (2" radius)
- Retrieves `f:clf_snnSnVsOthers_score` from Fink's SN classifier
- Results: 23/39 candidates got real scores, 10 false positives removed

#### 3. Optimized Observing Sequence

**Problem:** Targets sorted by merit/RA caused excessive telescope slewing between DDFs.

**Solution:** Added `optimize_observing_sequence()` in `core/magellan_planning.py`.

- Greedy nearest-neighbor TSP weighted by merit (0.6) and slew distance (0.4)
- Respects visibility windows (observes targets at optimal times)
- Generates scheduled UT times for each target

**Example (15 targets, 2026-03-12):**

- Total slew: 91° (vs ~400°+ with naive merit sort)
- Clusters observations by DDF field
- Sky map visualization with color gradient (start→end of night)

#### 4. Code Architecture Refactor

**New Modules:**
| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration dataclasses |
| `core/report.py` | `ReportGenerator` class for PDF generation |
| `utils/rubin.mplstyle` | Custom matplotlib style for LaTeX rendering |

**Configuration Classes:**

- `MeritConfig` — tau, mag_optimal, host weights, extinction scale
- `ObservatoryConfig` — Las Campanas location, airmass limits
- `BrokerConfig` — timeouts, tolerances, circuit breaker settings
- `PipelineConfig` — quality thresholds, exposure times
- `PathConfig` — cache, output, log directories

**Broker Client Standardization:**

- `FinkLSSTClient` now inherits from `BaseBrokerClient`
- Implemented abstract methods `query_alerts()`, `get_stamps()`
- Exported `AtlasClient` and `Alert` dataclass from `broker_clients`

### Report Improvements

- LaTeX formatting for scientific notation ($W_{\rm time}$, etc.)
- Fixed text overlapping in tables
- Consistent merit-based sorting throughout all pages
- Observing sequence sky map with slew path arrows

### Files Changed

```
+config.py                    — NEW: Centralized configuration
+core/report.py               — NEW: ReportGenerator class
+utils/rubin.mplstyle         — NEW: LaTeX matplotlib style
 core/magellan_planning.py    — optimize_observing_sequence(), compute_merit_breakdown()
 core/__init__.py             — Export ReportGenerator
 broker_clients/fink_client.py — Inherits BaseBrokerClient, get_classifications()
 broker_clients/__init__.py   — Export AtlasClient, Alert
 run_tonight.py               — Merit breakdown columns, sequence optimization
```

---

## 2026-03-12 — Environment Setup & Initial Pipeline Run

### Completed

- Fixed conda environment: Python 3.12 (3.14 broke dependencies)
- Installed `antares-client`, `psycopg2-binary`, configured RSP TAP token
- First successful end-to-end pipeline run on MJD 61111

### Initial Results

| Stage                   | Count |
| ----------------------- | ----- |
| Fink candidates         | 167   |
| ANTARES candidates      | 196   |
| ALeRCE-ZTF              | 4     |
| After merge/dedup       | 334   |
| With Rubin diaObjectIds | 85    |
| Final (successful fits) | 6     |

**Main bottleneck:** 56 candidates rejected for "too few high-SNR points" — Rubin DP1 has sparse early cadence.

### ANTARES Optimizations Added

- Parallel DDF field searches (ThreadPoolExecutor, 3 workers)
- 60-day date pre-filter (skips loci without recent activity)
- Persistent locus cache (`antares_locus_cache.json`)
- Wall-clock time reduced from ~30 min to ~7 min

### ANTARES Performance by Field

| Field    | Checked | Accepted | Notes                        |
| -------- | ------- | -------- | ---------------------------- |
| XMM-LSS  | 2000    | 0        | 95% rejected as old_activity |
| M49      | ~500    | ~2       | ZTF-heavy, old objects       |
| COSMOS   | 324     | 28       | Good                         |
| ECDFS    | 597     | 28       | Good                         |
| ELAIS-S1 | 121     | 28       | Southern, mostly Rubin       |
| EDFS_a/b | 34 each | 28 each  | Southern, mostly Rubin       |

---

## Code Architecture

```
run_tonight.py              — Main CLI, orchestrates alert pipeline
supernova_monitor.py        — Broker query coordination
config.py                   — Centralized configuration

core/
  alert_aggregator.py       — Merge/dedup across brokers
  peak_fitting.py           — Light curve fitting (parabola, Villar)
  magellan_planning.py      — Merit function, observability, scheduling
  report.py                 — PDF report generation

broker_clients/
  base_client.py            — Abstract base class
  fink_client.py            — Fink LSST API (inherits BaseBrokerClient)
  antares_client.py         — ANTARES cone search (parallel)
  alerce_client.py          — ALeRCE API + direct DB
  atlas_client.py           — ATLAS forced photometry (retry/backoff)
  tns_client.py             — TNS cross-match for duplicate detection

orchestrator/               — LLAMAS spectroscopic scheduling (MAGNETS)
  models.py                 — Target, ScheduledEntry, ObsPlan dataclasses
  normalize.py              — CSV/RubinAlerts loader, exposure estimation
  planner.py                — Greedy scheduler, twilight, airmass, standards
  output.py                 — Timeline, TCS catalog, summary writers
  config.py                 — LLAMASConfig (site, overhead, exposure table)
  cli.py                    — python -m orchestrator CLI

host_galaxy/
  morphology_filter.py      — Galaxy classification

utils/
  catalog_query.py          — SDSS/PS1/SkyMapper queries
  extinction.py             — Galactic E(B-V) from SFD
  rubin.mplstyle            — LaTeX matplotlib style

cache/
  alert_cache.py            — SQLite caching system
```

---

## Quick Reference

**Run pipeline:**

```bash
python run_tonight.py 61101 --min-prob 0.3 --days-back 30
```

**Output location:** `nights/ut{YYYYMMDD}/`

- `candidates.csv` — ranked candidate list with merit breakdown
- `report_ut{date}.pdf` — summary with light curves and sky map
- `magellan_plan.cat` — Magellan TCS format catalog
- `observing_schedule.txt` — human-readable schedule with merit breakdown
- `optimized_sequence.csv` — slew-minimized observing order
- `pipeline.log` — detailed execution log

**Key credentials:**

- RSP TAP: `~/.rsp_token`
- ATLAS: `~/.atlas_credentials` (register at https://fallingstar-data.com/forcedphot/)
- TNS: `~/.tns_credentials` (register at https://www.wis-tns.org/user)
- ALeRCE DB: hardcoded in `alerce_db_client.py`

---

## Status & Next Steps

### Completed — Alert Pipeline

- [x] Merit breakdown in reports
- [x] Fink enrichment for ANTARES candidates
- [x] Optimized observing sequence (slew-minimized)
- [x] Code architecture refactor (config.py, report.py)
- [x] LaTeX-formatted output
- [x] ANTARES parallel search & date pre-filter
- [x] Host morphology catalog queries (SDSS/PS1/SkyMapper)
- [x] E(B-V) extinction propagation fix
- [x] GLADE+ galaxy catalog integration
- [x] Increased morphology search radius (2 arcmin)
- [x] TNS cross-matching for duplicate detection
- [x] Nuclear offset filter for AGN/TDE rejection
- [x] ATLAS credential verification (integration complete)
- [x] ATLAS forced photometry (credentials working)
- [x] NED redshift queries with caching
- [x] SALT2/SALT3 template fitting (sncosmo)
- [x] ZTF photometry via ALeRCE (batch DB queries, 2" position cross-match)
- [x] Rise time constraints filter (rejects slow risers > 30 days)

### Completed — Spectroscopic Orchestration

- [x] Meeting with Yize (Ashley's group) on current workflow
- [x] Read Stubbs 2026B proposal — MAGNETS context, LLAMAS specs
- [x] Design doc for orchestration layer (`docs/design/spectroscopic-orchestration.md`)
- [x] Knowledge wiki updates
- [x] Phase 1: Normalizer, LLAMAS Planner, CLI (`orchestrator/` package)
- [x] Validated on DDF targets (99% efficiency) and Yize's real targets
- [x] All 5 broker clients verified working (Fink, ANTARES, ALeRCE, ATLAS, TNS)
- [x] ATLAS retry/backoff and token re-auth
- [x] Phase 2: Multi-program time accounting + composite prioritizer
- [x] Phase 3: RubinAlerts integration (candidates.csv → LLAMAS plan with budgets)
- [x] Phase 4: Nightly time reports + season progress reporting
- [x] Architecture document for design review (`docs/design/architecture.md`)

### Remaining — Alert Pipeline

- [ ] Historical validation on archived DP1 data (MJD 60630-60650)
- [ ] Unit tests with pytest coverage
- [ ] Direct RSP DiaObject photometry queries
- [ ] AGN/QSO catalog cross-match (Million Quasar Catalog)

### Remaining — Spectroscopic Orchestration (MAGNETS)

- [ ] Integration test with real alert pipeline candidates.csv
- [ ] Boyd et al. 2026 WD standard catalog (if available)
- [ ] Google Sheet ingester for manual target requests
- [ ] Multi-night look-ahead optimization

### Known Issues

1. **GLADE+ optical photometry sparse** — most entries have only WISE, morphology returns 'uncertain'
2. **Rubin cadence** — DP1 has sparse early data, limiting light curve quality
3. **TNS cross-match skipped in --fink-only mode** — need to enable in full multi-broker runs
