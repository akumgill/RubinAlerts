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

### 2026-08 storm: Rubin dark, ZTF pivot (IMPORTANT — diagnosed this session)
A Chilean storm took out BOTH Chilean facilities, and it reshaped the data path:
- **Rubin/LSST (Cerro Pachón) alert stream is dead — no data after 2026-07-14**
  (Fink statistics table; hard cliff from ~500k alerts/night to zero for 23
  nights). This is ~26 days *before* the planned 8/9 downtime — a storm, not the
  scheduled maintenance. The ~3,584 "Rubin"-tagged Fink alerts a run still pulls
  are a **stale rolling window**, not fresh sky.
- **ALeRCE (Chile-hosted) is also frozen at 2026-07-08** — connection/creds fine
  (a broad query returns 50k rows), not the dec cut (142 of 156 survive dec<=+22),
  just no new ZTF detections ingested. Under the 30-day default window it
  self-zeros ~8/8, exactly as the downtime begins. Leaning on a Chilean broker to
  cover a Chilean-weather outage is a single point of failure.
- **LCO / Magellan (Las Campanas) suspended too** — restart **Aug 11, local
  observations only** (Yuri Beletsky observing for us). Aug 4/7/10 nights are OFF;
  next real nights **Aug 13 + Aug 16, both LLAMAS full**. Yize asked for targets.

**Fix (shipped): Fink-ZTF as the live, out-of-region discovery ingress.**
- Probed `api.ztf.fink-portal.org`: **live to today** (~150k alerts/night), hosted
  in **France (IN2P3)** so independent of the Chilean outage; ZTF itself is at
  Palomar (CA), unaffected. Exposes the SN surface we need: `/api/v1/latests`
  (finkclass `Early SN Ia candidate` / `SN candidate` / `(TNS) SN Ia`), SNN+RF Ia
  scores, `/api/v1/objects` light curves, conesearch.
- New `broker_clients/fink_ztf_client.py` (`FinkZTFClient`): forks
  `FinkLSSTClient`'s `_post` transport, translates the ZTF schema (jd→mjd,
  fid→band, magpsf-is-mag not flux, `ZTF…` objectId strings, `/objects` not
  `/sources`+`/fp`, `/latests` not `/tags`, `d:snn_snia_vs_nonia` Ia score).
  `fetch_fresh_sn_candidates` applies the wide cuts. Unit-tested (schema + cuts).
- `scripts/generate_ztf_candidates.py`: writes a queue-schema candidate CSV for a
  night. First real run (Aug-13): **1213 raw → 117 after cuts** (fresh<=15d,
  dec<=+22, r<=21.5, Ia>=0.5); top 20 written to `ref/stubbs_llamas_2026-08-13.csv`
  as CfA-Stubbs / LLAMAS. Many are `(TNS) SN Ia` (already spectroscopically
  classified — see prior-spectroscopy note below).

### FinkZTF wired into run_tonight wide mode — DONE (apples-to-apples)
Only the top-of-funnel differs; everything downstream is the identical code.
- `fetch_finkztf_wide_candidates` (run_tonight.py) mirrors `fetch_ztf_wide_candidates`'s
  schema + screens (Galactic-plane, hostless brightness) but sources from
  `FinkZTFClient`. Fink's SNN Ia-vs-nonIa is a genuine P(Ia), so it populates
  `sn_ia_prob` directly. Added as a **third source in the same `merge_alerts` call**
  (`{'Fink', 'ALeRCE-ZTF', 'Fink-ZTF'}`), with a `Fink-ZTF` broker-status entry.
- `core/alert_aggregator.py`: `Fink-ZTF` added to the ddf_field + ztf_oid/sn_score
  fallback branches (merge-layer plumbing for the new broker; no science change).
- Verified end-to-end live: 116 Fink-ZTF candidates → same aggregator → coalesce →
  normalize → `mean_ia_prob` pooled, `effective_prob`, prob_source=ml; 101 after
  P(Ia)>=0.3. From here the LC-fit / merit / phase / moon / airmass / ranking are
  the exact same code the Rubin path runs. Tests: new `test_finkztf_only_stream_
  downtime` + patched wide-path tests; suite 312 passed.
- REMAINING: a full `run_tonight --sky-mode wide` end-to-end run for Aug-13/16 (LC
  fits over the network, minutes) to produce the ranked candidates.csv, vs the
  classifier-shortlist `scripts/generate_ztf_candidates.py` gives instantly.

### ZTF light-curve sourcing + photometry blackout guard — DONE
The first full wide run exposed a gap: discovery came from ZTF, but the LC-fit
stage still fetched photometry from Fink-LSST (Rubin, dark) → **0/348 light
curves**. Fixed so light curves come from wherever the object was seen:
- `fetch_finkztf_photometry_batch` pulls live ZTF light curves from Fink-ZTF
  `/api/v1/objects` (AB mag → nJy), keyed by ztf_oid. Wired into `fetch_and_fit`:
  Fink-ZTF preferred for any object carrying a ztf_oid, ALeRCE positional match
  as fallback, Fink-LSST for Rubin objects. Verified live: 3/3 known ZTF objects
  return real nJy light curves.
- `photometry_coverage` + a **PHOTOMETRY BLACKOUT** error: candidates present but
  0 light curves from ANY source is now a loud, greppable failure (a sourcing
  problem, not an empty sky), not a silent INFO log found by manual inspection.
  Tests (`tests/pipeline/test_photometry_sourcing.py`) assert the mag→nJy
  conversion, coverage counting, and that `fetch_and_fit` emits the blackout
  error. Suite 317 passed.

### Open / next
- **Prior-spectroscopy enrichment (asked 2026-08-06):** flag each candidate with
  TNS classification status (we already ingest TNS; the Fink `(TNS)` prefix is a
  free signal) + WISeREP epoch coverage (`n_spectra`, last phase, group). Decision
  support, not a hard cut: for a typing program "already classified" → deprioritize;
  for a spectral-time-series/cosmology program a single near-peak classification
  spectrum often doesn't satisfy the science, so re-observing at a new phase is
  still valuable.
- **Repoint the live site** off the cancelled Aug-7 to Aug-13 LLAMAS + mark
  Aug 4/7/10 storm-cancelled (Yuri Beletsky observer) — do *after* Aug-13
  candidates are in the queue so a visitor doesn't see a blank night.
- **RSP alerts API** (Chris's link, rsp.lsst.io/guides/api/alerts.html): a real
  queryable Rubin alert-packet API (needs the RSP token Akum now has) — a
  Fink-independent Rubin path for *when Rubin returns*; useless now (Rubin issuing
  nothing) and Rubin-only so not a ZTF substitute.
- LDSS3 native "click" catalog + finder charts, the ORACLE/Villar feed adapter,
  real post-night reconcile, and swapping the demo group credentials for real
  MAGNETS logins in the Render dashboard.

### S/N-based exposure ETC (Chris's curve) — core built
`core/snr_etc.py` digitizes Chris's "LLAMAS: exposure time to reach SNR=5 on SN Ia
peak vs redshift" curve (SNR=5 per pixel; indexed by peak apparent r). Two of his
calibration facts fold in: SNe have broad features so we bin ~n_bin≈10 px in
wavelength (√10≈3.2× SNR → net exposure ~10× shorter), and SNR∝√t (target scales
by (target/5)²). So `t_net(r) = t_curve_pp5(r) × (target_snr/5)² / n_bin`.
`split_exposure` breaks the net into 300–600 s sub-exposures for CR rejection.
Sample: r≈23.4 (z≈0.8) → ~8 min net → 2×300 s (why LLAMAS can chase faint SNe).
Tested (7). WIRED (2026-08): the ETC is now the PRIMARY tier of the orchestrator
cascade `estimate_llamas_exposure` (config `use_snr_etc=True`, LLAMAS-only per
Akum) — magnitude-driven, ahead of the proposal-table/mag-scaling fallback.
Config: `snr_target_binned=10` (binned typing S/N), `snr_binning=10`,
`snr_min_minutes=10` floor, `snr_moon_factor` (dark 1.0 / grey 1.4 / bright 2.0).
Floor set to 10 min (from 2→5→10) after a target-switch-overhead analysis: a
LLAMAS switch costs ~5 min in practice (overhead 1 + acquisition 2 + slew), so
a 10-min minimum keeps the visit ~2x the overhead (~70% science efficiency)
instead of churning overhead-dominated ~5-min visits. On Aug-13 this moved the
plan from 36 sched / 4.1 h to 31 sched / 6.1 h in a 10.3 h dark window.
Wired values (grey): mag<21 → 5 min floor; r=23.4 → ~44 min; r=24 → ~82 min.

>> OPEN QUESTIONS — MEETING PREP (as of 2026-08-07; Akum to raise w/ Chris this
   weekend + the group). Grouped by owner.

   @CHRIS (exposure calibration):
   - TARGET BINNED S/N for typing? The knob that DRIVES exposure. His note gives
     the curve's per-pixel S/N (5) + the ~10x binning gain but NOT the target
     binned S/N; we defaulted to 10 (t = 0.4x the curve; 5 marginal, 15 ≈
     curve-as-is, higher = cosmology/spectral-standardization grade). The Aug-13
     plan is time-rich (~6 of 10.3 h used), so we can afford higher S/N.
   - MINIMUM exposure per target on LLAMAS? Floored at 10 min (~2x the modelled
     ~5-min switch overhead). Yize's 30-min minimum is LDSS3 (slit overhead +
     characterization S/N) — don't just copy it. Is 10 min right for LLAMAS?
   - Confirm the true LLAMAS acquisition/switch OVERHEAD (sets the floor + the
     scheduler's per-target ops time).
   - Sanity-check the exposure model differs by TYPE (Ia broad-feature, bins 10x;
     II/IIn narrow-line, can't bin) and by INSTRUMENT (need an LDSS3 S/N curve;
     the ETC today is LLAMAS-only). Cross-check that held: Villar's LDSS3 30-min
     block at mag 20 ≈ our curve stripped of binning at S/N~20 (~37 min), so the
     calibration is consistent; the ~15x gap is binning + S/N target.

   @GROUP (policy forks that change how the tool behaves):
   - CHARGING RULE (parity-queue core): when a target is observed on a SHARED
     night (e.g. a Stubbs Ia on Yize's LLAMAS night), whose budget docks — the
     target's program or the observing PI's? Needs collaboration buy-in.
   - ONE SPECTRUM vs a TIME SERIES? If typing → one spectrum, then strike it. If
     cosmology time-series → multiple epochs, DON'T strike, track cumulative
     coverage. Changes both the ledger (below) and the merit/cadence logic.
   - What else must the tool SHOW to be useful to the collab?
   - Is the per-night output a useful DELIVERABLE for the observer?
   - How do we account for STORM-LOST nights (Aug 4/7/10 already marked
     cancelled; broader policy)?
   - How do we track what was ACTUALLY OBSERVED, to strike it from the future
     queue? Manual for now; a reconciliation ledger later (also feeds the
     allocations "used" bar, currently 0).

   @VILLAR (concrete data gap):
   - SANITY-CHECK the instrument of the list we have. We tagged it LDSS3 from the
     source-document TITLE (the "VTDA" list), NOT from Yize per-target. Evidence
     leans LDSS3 (30/60-min block exposures = slit/characterization; narrow-line
     II/IIn targets), but it's UNCONFIRMED and Yize has time on BOTH (15 h LDSS3
     + 30 h LLAMAS). Confirm: is this list LDSS3?
   - Yize's LLAMAS target list. His near-term nights (Aug 13/16) are LLAMAS and
     we have no LLAMAS picks from him — his LLAMAS night currently shows only
     Stubbs opportunistic fillers. Need the list or a feed. (Same conversation.)

   OPERATIONAL (Akum, not meeting topics):
   - Confirm the derived per-program allocations; swap demo logins for real
     per-group keys before the collab uses it live.
   - Add a 1 GB persistent disk (DB_PATH=/var/data/queue.db) right before real
     submissions — currently ephemeral (re-seeds from repo each deploy).

REMAINING:
- Wire the ETC into `estimate_exposure_minutes` (service/queue preview) and the
  pipeline's `estimate_exposure_time` (core.magellan_planning) too, for full
  consistency (the authoritative plan path — the orchestrator — is done).
- Consider a `w_z` redshift preference (peak 0.6-0.8) + possibly an
  expected-future-coverage (cosmology-LC) factor.
- ~~Precise pixel-digitization~~ DONE: curve extracted from the saved PNG
  (`docs/design/figures/llamas_snia_exptime_vs_z.png`) — y calibrated on the
  10-/60-min dotted refs, x→r via the top-axis ticks; cross-checks held (10-min
  crossing r=21.4, 60-min r=23.2). Caught ~2× eyeball errors at the bright end.
  Still TODO: a **host-background term** (curve is a point-source SN Ia peak calc
  — faint SNe on bright hosts run optimistic; mitigated for LLAMAS by IFU host
  subtraction, so second-order ~1.2–2×, not a blocker).
- **Science-scope decision (collaboration, not ours):** Chris wants to focus
  **0.6<z<0.8** (r≈23.4–24), which is OUTSIDE our current cuts (z≤0.4, r≤21.5) and
  in the curve's **extrapolated** (r>21) region — pursuing it means relaxing the
  selection cuts and accepting extrapolated exposures (flagged via the
  `extrapolated` return).

### SHIP BURNDOWN (2026-08-07) — tracking list, prioritized, with decisions
Two gates: (A) put in front of collaborators for a sandbox/validation round;
(B) rely on it for real observing.

TIER 0 — before collaborators touch it:
1. [Akum: WILL DO] Persistent disk on Render (1 GB, DB_PATH=/var/data/queue.db)
   — else submissions vanish on redeploy.
2. Credentials. DECISION: the GROUPS_JSON env-var approach IS sufficient at this
   scale (one key+password per program). Work is operational — strong keys
   (generated), keep secret in Render env, distribute per group — NOT code.
   Optional hardening: fail-closed if GROUPS_JSON is unset in prod so it can't
   silently fall back to the public demo keys. Known limits: group-level (not
   per-person) logins; key rotation = edit env + redeploy.
3. [DONE] Initial queue = empty + Villar only. Seed drops the Stubbs Aug-13 ZTF
   fills; seeds just the Villar LDSS3 standing list; other groups populate via
   API/UI. (Villar instrument still unconfirmed — see @Villar below.)
4. [DONE] Getting-started for collaborators: a 30-sec quickstart at the top of
   the API guide + a prominent "New here? Start with the guide" callout on the
   login page.

TIER 1 — before real observing:
5. Observed-tracking: MANUAL for now (decision) — revisit once we see the
   observer's per-night output, then build a reconciliation ledger.
6. Charging rule DECIDED (Akum, needs group ratification): charge a group for the
   OBSERVED time (exposure + overhead) of ITS OWN enqueued targets, regardless of
   whose night it is (if C group's targets fill my night, C is charged). Matches
   the existing accounting (budget factor keyed by the target's program); feeds
   the allocation "used" bars once reconciliation (#5) exists.
7. [Villar: WILL DO] Confirm the list's instrument (LDSS3 vs LLAMAS) + get Yize's
   LLAMAS list.
8. [Chris: WILL DO] Exposure answers (target S/N, min exposure, switch overhead).
   NOT a launch blocker — sensible defaults now.

TIER 2 — quality/tuning, post-launch:
9. Wire the ETC into the queue-preview estimator (api.service.estimate_exposure_
   minutes) — the ONLY remaining wiring (the authoritative orchestrator path is
   done; the pipeline's magellan_planning estimator is retired). Low effort.
10. LDSS3 S/N curve — PARKED: no LDSS3 calibration data (can't build now) AND it
    doesn't matter for the current programs (Stubbs/the Ia+ETC program is
    LLAMAS-only; the LDSS3 users do narrow-line characterization the Ia-binning
    ETC doesn't model). LDSS3 keeps the old cascade. Host-background term also
    deferred (second-order; LLAMAS IFU subtracts host locally).
11. Cold-load latency. DONE: revision-keyed dashboard cache + (a) vectorized
    airmass transform + (b) background cache-warm the default night at startup —
    so reads are warm except the first hit of a never-viewed night or the first
    read right after a write. PARKED (the "true" fix, add if post-submit lag
    annoys people in the validation round): EAGER RECOMPUTE-ON-WRITE — the write
    endpoints kick a background recompute of the HOT cache keys (viewed nights +
    default) so readers stay warm even right after a submit. Read-heavy /
    write-light tool, so moving the cost to write-time is the right trade. It
    HIDES but doesn't REDUCE the cost; the underlying ~10 s is the greedy
    scheduler (run_nightly, ~O(N^2) per-slot pick) — a real algorithm change,
    only worth it if reads must be cheap with no background work.
12. Observer polish: finder charts (annotated sky cutouts per target to confirm
    pointing) + LDSS3 native "click" catalog (LDSS3-GUI-specific target-list
    format; we ship a generic catalog export today). Partly blocked on the exact
    LDSS3 GUI format.

VALIDATION GATE: run the sandbox "fake night" round with a group before real
reliance. Critical path to a first invite: #1 -> #2 -> #3 -> #4 -> sandbox.
