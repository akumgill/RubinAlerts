# Design Review Loop — Change Log

A three-role loop reviewing and improving the RubinAlerts design end-to-end:

- **Reviewer ("Chris")** — critical reviewer. Mandate: no wasted time on sky,
  transparent and intelligent prioritization. Produces findings.
- **Architect** — takes reviewer findings, pushes back where warranted, produces
  a sequenced implementation plan.
- **Engineer** — implements the agreed plan with tests, prunes dead code, avoids
  hammering external services/DBs.

Scope (per user, 2026-06-23): **whole design** (ingestion of both streams, merit
scoring, prioritization, scheduling, time accounting, outputs/transparency) and
**implement all agreed items** with frequent commits.

Branch: `design-review-loop`

---

## Timeline

### 2026-06-23 — Loop initialized
- Created branch `design-review-loop` off `main`.
- Scaffolded this log.
- Committed pre-existing in-progress edits (CLAUDE.md, docs/rubinalerts_pipeline.tex,
  untracked review .tex/.pdf) as a baseline so subsequent commits are isolated.

### 2026-06-23 — Stage 1: Reviewer ("Chris") findings

Read-only critique of the whole design. 18 findings; Top-5 must-fix in **bold**.

| ID | Sev | Area | Finding (short) |
|----|-----|------|-----------------|
| **R1** | Critical | merit | Moon penalty computed but never folded into ranking merit; report falsely shows W_moon applied |
| **R2** | Critical | prioritization | Composite 100/20/50 mix arbitrary; phase weight can demote P1 below P4; undocumented |
| **R3** | High | accounting | `reconcile` default state path ≠ `run-nightly` write path → silent double-charge |
| R4 | High | accounting | Charge-on-schedule uses padded wall-clock (gap-fill/end-of-night), not science time |
| **R5** | High | ingestion | No consolidated broker-liveness signal; silent failure looks like "no SNe" |
| **R6** | High | ingestion | Southern DDFs single-broker by construction, yet num_brokers boosts merit (geographic bias) |
| **R7** | High | merit/ingestion | ANTARES P(Ia) proxy (uncalibrated, capped 0.50) averaged with ML prob, drives w_prob |
| R8 | Med | prioritization | merit→P1-P4 quartile mapping is relative-only; meaningless across nights / small lists |
| R9 | Med | scheduling | Greedy scheduler ignores slew between picks; 1-min overhead assumes negligible slew |
| R10 | Med | scheduling | Standard stars start/end only; spec calls for mid-night interleaving |
| R11 | Med | accounting | budget_factor thresholds global across moon phases; 5h cliff not scaled to program size |
| R12 | Med | merit | Exposure cascade has redshift-table cliffs; two subsystems disagree on exposure model |
| R13 | Med | ingestion | Dedup uses RA/Dec box, not angular separation; dec-dependent radius hurts southern fields |
| R14 | Med | transparency | Orchestrator output omits score components; PI can't reconstruct ranking |
| R15 | Med | ingestion | Manual PI-queue: CSV has no program/budget; defaults to phase=1.0 (treated near-peak) |
| R16 | Low | scheduling | 900s sub-exposure split hardcoded/undocumented; rounds down |
| R17 | Low | code-health | config.py docstring "Magellan/Clay" → should be Baade |
| R18 | Low | transparency | Fallback scheduler scale differs from prioritizer; mode not stamped in output |

**New operational constraints from PI (2026-06-23), fed into architect:**
- **Rubin offline 2026-08-09 → end of Aug 2026** — ZTF-fed brokers (ALeRCE-ZTF, ANTARES)
  become the *primary* alert source in that window. Raises priority of R5 (liveness) and
  R6 (single-broker/coverage handling). Southern DDFs lose their only source during downtime.
- **2026B observing nights** (LLAMAS unless noted): Chris/us 12/15 & 1/12 (full);
  Yize/Villar 8/10, 8/13, 8/16 (full), 9/6, 9/7, 10/3 (half); Conor Ransome LLAMAS
  8/4, 11/2, 12/8 + LDSS3 8/7, 9/12, 10/5. (More observers likely TBD.)

### 2026-06-23 — Stage 2: Architect response + plan

Validated all findings against code. Notable pushback / corrections:
- **R12** corrected: `2.5^dmag` ≡ `10^(0.4·dmag)` (identical) — that sub-claim void. Real
  issue is the discrete redshift-table cliffs + omitted moon/airmass. "Unify on one
  exposure model" **rejected** (subsystems have different inputs: z-driven vs mag-driven).
- **R8** absolute merit thresholds **deferred** (merit not cross-night calibrated) → guard
  <4-target case + document P-labels are within-night relative.
- **R15** "require explicit program" / build Sheet ingester **rejected** (breaking /
  out-of-scope) → honor optional columns with soft default + warning.
- **R9** full path optimization & **R18** single scoring function **rejected** as
  over-engineering for clustered DDF targets → modest slew penalty + mode stamp.
- **R14** merged into R2 (transparency half).
- **R1** found *worse*: `candidates.csv` (orchestrator's input) carries moon-free merit
  with no moon_penalty column at all.
- **R5** promoted to highest ingestion priority given Rubin downtime.

**Sequenced plan (engineer executes top-to-bottom; every test runs offline w/ fixtures):**

| Chunk | Items | Resolves | Commit theme |
|-------|-------|----------|--------------|
| A | W0 scaffold pytest + fixtures | — | test scaffold |
| B | W1 fold moon into merit; W2 coverage-aware w_broker | R1, R6 | merit correctness |
| C | W3 broker-liveness status; W4 ML-only w_prob + angular-sep dedup | R5, R7, R13 | ingestion safety |
| D | W5 unify state path; W6 charge science-time + phase-aware budget_factor | R3, R4, R11 | accounting |
| E | W7 normalized/documented score + breakdown; W8 slew penalty + mode stamp + relative-P guard | R2, R8, R9, R14, R18 | prioritizer/scheduler |
| F | W9 interpolate exposure table + mid-night standards | R10, R12 | exposure/standards |
| G | W10 config/doc cleanups + manual-target columns | R15, R16, R17 | chore |

Hard rule: **no test may hit a live broker/DB/API** — mock/stub everything.

---

## Stage 3: Engineer — execution record (all chunks complete)

Test suite grew 0 → **74 tests**, all green, fully offline (mocked brokers/DBs, temp-file
JSON). Commits on `design-review-loop`:

| Commit | Chunk | Resolves | Summary |
|--------|-------|----------|---------|
| `20c1dad` | A | — | pytest scaffold + fixtures (no live services) |
| `882e222` | B | R1, R6 | fold moon penalty into ranking merit (single source of truth); coverage-aware `w_broker` so southern single-broker DDFs aren't geographically penalized |
| `f53dfb2` | C | R5, R7, R13 | consolidated broker-liveness status (report + `broker_status.json` sidecar); ML-only `mean_ia_prob` (ANTARES proxy split out); angular-separation dedup |
| `9013d3c` | D | R3, R4, R11 | unified state-file path (`_default_state_path`, no reconcile double-charge); charge science-time only (`ScheduledEntry.charged_minutes`/`padding_minutes`); phase-aware fractional `budget_factor` |
| `215b12d` | E | R2, R8, R9, R14, R18 | documented/normalized composite score (named constants, phase clamped [0,1]); `(score, breakdown)` returned + persisted (`score_breakdown.json`, summary table); slew penalty; scoring-mode stamp; <4-target P-label guard + relative-P caveat |
| `efa3e36` | F | R10, R12 | interpolate redshift exposure table (no cliffs); cadence-based mid-night standard insertion |
| `bd8fdb5` | G | R15, R16, R17 | Baade docstring; config-ize 900s split + 10s rounding; honor manual-target `program`/`phase_weight`/`peak_mjd` columns (soft default + warning); architecture.md updates |
| `24a410a` | W11 core | new (PI req) | `orchestrator/target_ledger.py`: coordinate-keyed cross-night integration ledger + completeness factor folded into the score |
| `7057aca` | W11 wiring | new (PI req) | wire ledger into run-nightly (schedule only *remaining* time, exclude satisfied targets), planner charge, and CLI (`--target-ledger`, `reconcile-target`, `ledger`) |

**Architect pushback honored** (not implemented, by design): unify exposure models (R12 —
forms were identical); absolute merit thresholds (R8 — deferred, merit not cross-night
calibrated); hard-require program / build Sheet ingester (R15 — out of scope); full path
optimization & single scoring function (R9/R18 — over-engineering for clustered DDFs).

**Verification:** `pytest` 74 passed; `python -m orchestrator --help` shows the new
subcommands; `run_tonight` imports clean. **No single-night regression** — every new
parameter defaults to today's behavior when no ledger/breakdown exists.

**Notes / optional follow-ups:** broker-liveness not yet surfaced through
`supernova_monitor.run_full_pipeline` (only the `run_tonight.py` path); ledger summary
shows latest-night required only; standards remain unbilled to science budgets (matches
prior start/end behavior). Production cross-broker dedup radius left at 1.0″ (tests use 1.5″).

---

## Follow-on features (post-loop, PI-requested 2026-06-23)

Chris's point: the most valuable phase to observe a SN depends on the program's science
(cosmology/standardization → peak; progenitor/CSM/exotic → rise). Built as two commits
(suite 74 → **96 tests**, all offline):

| Commit | Feature |
|--------|---------|
| `996abbc` | **Per-program phase preference** — `ProgramAllocation.phase_preference` (`peak`/`rising`); phase factor `exp(-((Δt−Δt_pref)²)/2τ²)`, Δt_pref peak=0 / rising=−7 d, τ=10 d. Signed `Δt` threaded from candidates.csv `delta_t` and manual `peak_mjd`. `get_phase_preference()`; example allocations updated (Stubbs=peak, Villar=rising). |
| `bcef109` | **Phase-split ledger + multi-group alert** — ledger cumulative split by observed-phase bucket (rising/peak/declining/all, ±5 d window); completeness judged per tonight's bucket so "done at peak" ≠ done on the rise. Old scalar ledgers auto-migrate. Multi-group alert (`multi_group_alerts.json` + summary + warning, with `same_phase` flag) when one object is wanted by >1 program. CLI `ledger` shows per-phase; `reconcile-target --phase`. |

**Doc regeneration:** `docs/design/spectroscopic-orchestration-review.tex/.pdf` rebuilt from
scratch (the Apr-2026 draft was made with a much earlier model) to reflect the current
system — all R-fixes, W11 ledger, and these phase features. Compiles clean (8 pp).

**Tunable / science-policy (decide with Chris + Ashley's group):** rising offset (−7 d), τ,
phase-bucket window (±5 d); multi-group conflict *arbitration* deferred (currently alerts only).



---

## 2026-07-13/14 — Wide-sky sprint + MAGNETS meeting prep

Status snapshot written 2026-07-14 (meeting day). ~30 commits
`8fd00e3..3ffd717`, suite 96 → **288 tests** (all offline-safe).

### The pivot: DDF-only → wide-sky selection

The DDF-centric funnel produced 8 faint unusable targets (77 min runtime);
inverted to payload-level wide-sky selection: r ≤ 21.5, z ≤ 0.4 (hostless
≤ 20.5), dec ≤ +22 (airmass 1.6 at LCO), fresh ≤ 30 d, baseline ≤ 150 d,
|b| ≥ 10, GCVS/VSX/Simbad-AGN/Gaia-parallax screens — all BEFORE per-object
work. Canonical demonstration night (`nights/wide9/ut20260713`): funnel
3584 Fink + 134 ALeRCE-ZTF → 157 fitted → **30 ranked finalists, 11 min
runtime**; 6/6 pre-classified finalists consistent with pipeline evidence
(rank 1 = confirmed SN Ia, 10-min target).

### Shipped since the June loop

| Area | What |
|------|------|
| Sources | ALeRCE-ZTF live SQL (`query_fresh_sn_candidates`, time-filtered, fixes arbitrary-slice bug); Fink payload selection; TNS daily-dump cross-match (1 download/night, 202k objects); NED timeout fix |
| Aggregator | Wide mode uses real `merge_alerts` (2″), PASSTHROUGH_COLUMNS reducers, cross-survey agreement stats |
| Probability | `effective_prob` chain (mean_ia_prob → sn_score → antares_proxy) + `prob_source` + `needs_classification`; `w_iaspec` [0.8–1.2] with positive-Ia ≥ neutral |
| Fitting | Survey-aware SALT2 (`salt2-extended` + F99), `salt_z_policy` (spec-z fixed / photo-z bounded / free box), `choose_best_fit`, tiered rescue; **multi-type template tournament in production** (`fit_template`/`run_template_tournament`, nugent Ibc/IIP/IIn; Step 5b enrichment on finalists; ground truth 6/6: 4 Ia→Ia, TDE→IIn, SN II→IIP) |
| Ranking | **Merit-per-hour** (single ranking; `merit_rate = merit × (45/exp)^0.5`, `--rank-alpha`); merit stays pure science value; per-program RankingProfile (IA + exotic strawman) |
| Scheduling | **Single authority**: pipeline RANKS, orchestrator SCHEDULES (run_tonight Step 9 → `orchestrator.run_nightly` → `nights/…/llamas/`); pipeline's own schedule outputs retired. Standards in-plan, slew+acquisition wall-clock (unbilled ops), exposure-density bonus, prospective duration-aware fairness with feasibility band (`fairness_tolerance`), shared-ops proration by science share |
| Cross-night | Per-target integration ledger (2″ coordinate-keyed, phase buckets, satisfied ≥ 0.95, frac>0 guard), multi-group alerts, mandatory reservation pass |
| Enrichment | Post-ranking finalist z-enrichment (TNS dump → NED → fixed-z SALT refits); template-tournament typing columns |
| Reporting | PDF report consistent with single ranking (merit/hr table, rank-order breakdown, updated merit reference, **LLAMAS observing-plan page** with per-program charges + ID→TNS-name legend) |
| Robustness | Fink breaker pause+resume; pandas≥3 dtype fixes; np.float64 SQL-param fix |

### Broker audits (recorded in memory + CLAUDE.md)

- **RSP/TAP**: static DP1 only; live APDB USDF-restricted → Fink is the sole
  live Rubin stream; redundancy = ZTF brokers.
- **ANTARES (2026-07-14)**: raw ES queries work (range on freshness/mag/dec;
  term queries ~0.2 s/object; knows 24/24 wide9 finalists). Wide harvest
  unwieldy (>4000 unfiltered young loci). Distinct payload (desoto typing,
  anomaly scores) sparse on young objects. **Parked**; per-finalist
  term-query enrichment is the cheap option if wanted.
- **ALeRCE classifier migration (found via ANTARES sanity check)**: legacy
  `lc_classifier` (149 objects, 75% TNS-reported) vs BHRF forced-phot
  (222, 92%) — overlap only 58. **Shipped multi-classifier union**
  (`query_fresh_sn_candidates_multi`, BHRF priority + legacy, provenance
  in `alerce_classifier`, no cross-classifier prob mixing): 313 objects
  (+110%). Legacy-only slice is 60% TNS-reported → union, not swap.
  ATAT(beta) (531 @ 64%) excluded pending calibration.

### Meeting materials

- Briefing artifact (two tabs: intro + technical appendix; collapsible
  sections; clickable per-target light curves; two-source architecture +
  funnel SVGs; Option-1/Option-2 operating scenarios; 7 discussion asks).
- `nights/wide9/ut20260713/report_ut20260713.pdf` — self-contained nightly
  report incl. executable LLAMAS plan.
- Fortino et al. (arXiv 2607.03532) ingested: typing intact at R=50/SNR=5 →
  "reconnaissance then commit" concept in asks (pending LLAMAS quicklook
  latency — question for Simcoe).

### Open (post-meeting queue)

TNS-fresh third source (typed SNe from the nightly dump; the ONLY source
for dec < −32 during the August Rubin downtime — 9 typed SNe there tonight);
ATAT calibration check; exotic top-of-funnel; tournament verdict → merit /
needs_classification + non-Ia rescue tier; real MAGNETS allocations
(example file still in use); epoch-cadence policy; S/N-based exposure/ETC;
ANTARES wide mode (parked); Google-Sheet manual front end.

### wide10 — first union-fed night (2026-07-14, MJD 61236)

Completed same day, ~15 min runtime despite 2x fit load:
union 313 → 287 after screens (220 BHRF + 67 legacy) + Fink → 310 fitted →
**63 ranked finalists (vs 30 on wide9)**. Finalist provenance: 48 BHRF /
9 legacy / 6 Fink-Rubin. Ground truth is rich: 61/63 TNS-reported, 22
spec-classified (20 Ia). z-enrichment gained 26 (22 TNS spec-z). Template
tournament (production, all 63): Ia=53, Ibc=8, IIP=1, IIn=1. Plan: 17
science targets + 4 standards, 9.7 h.

**Rank 1 = SN 2026reu** — the confirmed Ia (r=18.4, +4.5 d) that the
legacy classifier missed this morning and that motivated the union: found
via BHRF at 2 pm, implemented by 3 pm, top of the executable plan by 3:40.
Merit-per-hour 0.83 at an 11-min exposure; ranks 2–3 are two more
confirmed Ia at 18.4–18.6 (13/12 min). The rising Rubin photo-z object
(wide9's rank 2) sits at rank 4, now Δt = −4.1 d and template-tournament
preferred as Ia.

---

## 2026-08 — Collaboration tooling + deployment sprint

Shift from the alert pipeline (which the July sprint matured) to the
*shared-queue service* the MAGNETS collaboration adopted at the July meeting.
Akum owns the target-submission API; the pipeline is now one of several sources
feeding a shared scheduler.

### Target-submission API + queue service
- `api/` — framework-agnostic `TargetQueueService`: submit(upsert, 2" canonical
  dedup) / list / patch / withdraw / queue-summary / plan-preview. Bearer-key →
  program. `scheduler_bridge` runs the real orchestrator as a dry-run over the
  live queue (pipeline RANKS, orchestrator SCHEDULES — one authority).
- Interface spec circulated as an artifact; parity-queue prioritization (NOT the
  meeting's token idea — charge observed time, prioritize by parity bucket ×
  budget); P0 = the "observe tonight" guarantee folded into the priority enum.
- **Per-target `instrument` field** (LLAMAS | LDSS3 | EITHER): LDSS3 and LLAMAS
  schedule + budget as **two parallel systems**; `plan_preview(instrument=…)`
  filters to one universe and applies its overhead (LDSS3 ~10-min slit vs 1-min
  IFU).

### Budgets, from the real schedule
- Per-PI, **per-instrument** budgets derived from the nights spreadsheet
  (`ref/allocations_{LLAMAS,LDSS3}_2026B.yaml`). **Total-per-program budget**
  (`LLAMASConfig.budget_phase_aware=False`) — the earlier moon-phase-bucketed
  factor inflated whichever phase a PI's nights landed on (a PI with two grey
  nights looked like "20 h of grey"); moon still drives feasibility + w_moon,
  just not budget. Stubbs (Ia) = 20 h LLAMAS; UA = 30 h LDSS3 / 15 h LLAMAS;
  Villar = 15 h LDSS3 / 30 h LLAMAS.

### Aug 7 LDSS3 (first night) — dry runs
- Two-group cross-hash proven, then re-run through the service with real LDSS3
  budgets + instrument routing (10 real Villar/Dong targets + a UA stand-in).
  The Villar VTDA list is a **general semester-26B LDSS3 queue** (no per-target
  dates) → seed as standing entries, `ref/seed_villar_ldss3.csv` (Villar only;
  the Arizona list was a POC fabrication).
- Insight logged: a P3 can be scheduled ahead of a P2 (airmass window, not
  priority); and a target can bench for observability (short window), not budget
  — the total-budget re-run left the Aug 7 schedule unchanged, disproving an
  earlier "it's the budget" read.

### USDF audit (Akum got access)
- Probed via TAP (`~/.usdf_token`, works programmatically now): exposes only
  **dp1** (real ComCam, frozen 2024-12-12) + **dp02_dc2** (simulation). **No
  live prompt-products / APDB.** So USDF = static data, NOT a live alert feed;
  the incoming pipeline is unaffected. Corrected an earlier over-claim that USDF
  might unlock live alerts. (`rsp-tap-static-only` memory updated.)

### Tests + deployment
- Suite split into two buckets for two audiences: `tests/pipeline/` (Ia
  candidate generation — brokers, fitting, merit, selection; 170) and
  `tests/orchestrator/` (the scheduler given inputs — prioritizer, budgets,
  ledger, standards, normalize, API; 135). `test_smoke` at root. 308 total.
- **Website deployment** (in progress): `docs/design/deployment-plan.md` — one
  FastAPI process (JSON API + dashboard + browser login), SQLite, orchestrator
  in-process, on Render; dual auth (bearer key OR session cookie), writes scoped
  per-program. Build split across two parallel agents (backend: SQLite + auth +
  `/v1/dashboard` + Dockerfile + render.yaml; frontend: the live interactive web
  UI under `web/`).
- **Plan export** (`api/plan_export.py`): observer-facing outputs from the plan —
  the instrument catalog (the GUI-loadable file that avoids fat-finger entry),
  a CSV observing sheet, and a printable text sheet. (Browser print → PDF rather
  than a bundled PDF lib.)

### Integration pass — DONE (b750476, pushed to fork/design-review-loop)
Both parallel agents landed; merged and shipped:
- Seed rewritten to the in-repo Villar CSV (`api/seed.py` → `ref/seed_villar_ldss3.csv`),
  no `~/Downloads` / `seed_data.json` dependency; UA seeded only if they hold a key.
- `/healthz` (unauthenticated) added and `render.yaml` health check pointed at it
  — the old `/v1/dashboard` probe returned 401 and would have failed the deploy.
- `/v1/dashboard` and `/v1/plan/export` default to the seeded Aug-7 **LDSS3** night.
- Export endpoint wired to `api/plan_export.py` (catalog/csv/text) + three download
  links in the observing-plan header (same-origin GET carries the session cookie).
- `web/sample.json` regenerated from the live dashboard (Villar-only, program name
  reconciled to `CfA-Villar`); `data/` + `target_ledger.json` gitignored.
- **Live smoke test passed**: login (cookie) → dashboard shows exactly the 10 real
  Villar targets on LDSS3 → submit as UA (bearer + cookie, write-scoped, shows up in
  the shared queue) → all three exports return correct Content-Disposition. Suite
  308 passed / 1 skipped. Render builds from the pushed branch.

### Open / next
LDSS3 native "click" catalog + finder charts, the ORACLE/Villar feed adapter, S/N
ETC (Chris's curve; focus 0.6<z<0.8, a Rubin-era mode our current cuts exclude),
real post-night reconcile, and swapping the demo group credentials for real MAGNETS
logins (`GROUPS_JSON` / `*_KEY` / `*_PASSWORD` in the Render dashboard).
