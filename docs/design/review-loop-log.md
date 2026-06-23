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


