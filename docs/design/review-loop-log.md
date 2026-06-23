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

