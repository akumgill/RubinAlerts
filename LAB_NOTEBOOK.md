# RubinAlerts Lab Notebook

## Project Overview

Automated SN Ia candidate identification pipeline for Rubin LSST Deep Drilling Fields. Aggregates alerts from multiple brokers (Fink, ANTARES, ALeRCE), fits light curves, computes spectroscopic follow-up merit scores, and generates Magellan observing plans.

---

## 2026-03-12 — Environment Setup & Initial Pipeline Run

### Completed
- Fixed conda environment: Python 3.12 (3.14 broke dependencies)
- Installed `antares-client`, `psycopg2-binary`, configured RSP TAP token
- First successful end-to-end pipeline run on MJD 61111

### Initial Results
| Stage | Count |
|-------|-------|
| Fink candidates | 167 |
| ANTARES candidates | 196 |
| ALeRCE-ZTF | 4 |
| After merge/dedup | 334 |
| With Rubin diaObjectIds | 85 |
| Final (successful fits) | 6 |

**Main bottleneck:** 56 candidates rejected for "too few high-SNR points" — Rubin DP1 has sparse early cadence.

### Changes Made
- Added configurable quality cuts: `--min-snr-points`, `--min-bands`, `--min-fit-bands`
- Enhanced merit function with P(Ia) and host morphology weights

---

## 2026-03-13 — ANTARES Optimization & Diagnostics

### Completed
- Implemented parallel DDF field searches (ThreadPoolExecutor, 3 workers)
- Added 60-day date pre-filter (skips loci without recent activity)
- Added persistent locus cache (`antares_locus_cache.json`)
- Configured ATLAS credentials (`~/.atlas_credentials`)
- Fixed host morphology catalog queries (SDSS SQL, Pan-STARRS VizieR, SkyMapper)

### ANTARES Performance (with optimizations)
| Field | Checked | Accepted | Notes |
|-------|---------|----------|-------|
| XMM-LSS | 2000 | 0 | 95% rejected as old_activity |
| M49 | ~500 | ~2 | ZTF-heavy, old objects |
| COSMOS | 324 | 28 | Good |
| ECDFS | 597 | 28 | Good |
| ELAIS-S1 | 121 | 28 | Southern, mostly Rubin |
| EDFS_a/b | 34 each | 28 each | Southern, mostly Rubin |

The date pre-filter is working correctly — XMM-LSS is flooded with stale ZTF objects (no activity in 60 days). Parallel search reduces wall-clock time from ~30 min to ~7 min.

---

## Current Pipeline Issues (Fresh Analysis)

### 1. Host Morphology Not Populating
**Symptom:** All candidates have `host_morphology = "unknown"` in output CSV.

**Root cause:** The `MorphologyFilter.classify_host_galaxy()` method exists but isn't being called during the main pipeline run. The catalog query infrastructure (SDSS/PS1/SkyMapper) is implemented but not wired into the fitting loop.

**Impact:** Merit function morphology weight defaults to 0.7 for all candidates — no discrimination between elliptical (high Ia confidence) and spiral hosts.

### 2. ia_prob Uniformly 0.30
**Symptom:** Most candidates have `mean_ia_prob = 0.30` (the threshold value).

**Root cause:** ANTARES doesn't provide classification probabilities — it just passes/fails quality cuts. Fink provides `sn_ia_score` but it's not being propagated through merge/aggregation properly.

**Impact:** P(Ia) weight in merit function is essentially constant — no classifier-based prioritization.

### 3. No Auxiliary Photometry
**Symptom:** All candidates have `n_ztf = 0`, `n_atlas = 0`.

**Root cause:** ZTF photometry fetch requires ALeRCE API calls that aren't happening for ANTARES/Fink candidates. ATLAS was just configured today — hasn't been used in a run yet.

**Impact:** Light curve fits rely solely on sparse Rubin data. Missing pre-explosion history and independent color information.

### 4. Merit Function Effectively Simplified
Given issues 1-3, the merit function reduces to:
```
merit ≈ exp(-delta_t²/200) × exp(-((mag-20.5)/1.25)²) × 0.3 × 0.7
```
This is just time-to-peak × brightness — the classifier and host morphology discrimination are not contributing.

---

## Priority Queue (Re-evaluated)

### HIGH Priority
1. **Wire up morphology classification in pipeline**
   - Call `MorphologyFilter.classify_host_galaxy()` for each candidate after merge
   - Store result in DataFrame before merit calculation
   - ~30 min fix, high impact on ranking quality

2. **Fix ia_prob propagation from Fink**
   - Fink provides `d:rf_snia_vs_nonia` score — verify it's being captured as `sn_score`
   - Ensure merge/aggregation preserves this value
   - Cross-match ANTARES candidates against Fink to get scores

### MEDIUM Priority
3. **Run pipeline with ATLAS enabled**
   - Credentials configured, should work automatically for bright candidates
   - Will add pre-explosion limits and independent photometry

4. **Add multi-broker agreement factor to merit**
   - If Fink AND ANTARES both flag a candidate, boost confidence
   - Simple: `n_brokers` column already exists, add weight

5. **Add galactic extinction penalty to merit**
   - E(B-V) already computed — lower extinction = cleaner photometry
   - Weight: `exp(-E(B-V) / 0.1)` or similar

### LOW Priority
6. **Historical DP1 validation** (MJD 60630-60650)
   - Stress-test RSP TAP integration
   - Not blocking current operations

7. **Fink batch pre-filtering**
   - `--prefilter-min-sources N` implemented but less critical now
   - ANTARES date filter is more effective

### DROPPED
- ~~Clear ANTARES cache~~ — date pre-filter handles stale objects
- ~~Increase ANTARES cap beyond 2000~~ — diminishing returns, XMM-LSS has no recent transients anyway

---

## Code Architecture Notes

```
run_tonight.py          — Main CLI, orchestrates pipeline
supernova_monitor.py    — Broker query coordination
core/
  alert_aggregator.py   — Merge/dedup across brokers
  peak_fitting.py       — Light curve fitting (Bazin, Villar)
  magellan_planning.py  — Merit function, observability, catalog output
broker_clients/
  fink_client.py        — Fink LSST API
  antares_client.py     — ANTARES cone search (now parallel)
  alerce_client.py      — ALeRCE API + direct DB
  atlas_client.py       — ATLAS forced photometry
host_galaxy/
  morphology_filter.py  — Galaxy classification (NOT WIRED IN)
utils/
  catalog_query.py      — SDSS/PS1/SkyMapper queries
  extinction.py         — Galactic E(B-V) from SFD
```

---

## Quick Reference

**Run pipeline:**
```bash
python3 run_tonight.py 61111 --min-snr-points 3 --min-fit-bands 1
```

**Output location:** `nights/ut{YYYYMMDD}/`
- `candidates.csv` — ranked candidate list
- `report_ut{date}.pdf` — summary with light curves
- `magellan_plan.cat` — Magellan TCS format catalog
- `pipeline.log` — detailed execution log

**Key credentials:**
- RSP TAP: `~/.rsp_token`
- ATLAS: `~/.atlas_credentials`
- ALeRCE DB: hardcoded in `alerce_db_client.py`

---

## 2026-03-13 — Code Architecture Refactor & New Features

### Completed

#### 1. Centralized Configuration (`config.py`)
Created a new configuration module consolidating all magic numbers and thresholds:
- `MeritConfig` — merit function parameters (tau, mag_optimal, host weights)
- `ObservatoryConfig` — Las Campanas observatory parameters
- `BrokerConfig` — query parameters, timeouts, tolerances
- `PipelineConfig` — processing thresholds and quality cuts
- `PathConfig` — default paths for data and outputs

#### 2. PDF Report Module (`core/report.py`)
Extracted report generation from `run_tonight.py` into a standalone `ReportGenerator` class:
- Modular page generation methods
- LaTeX-compatible formatting for scientific notation
- Merit function reference page with parameter definitions
- Observing sequence sky map with slew visualization
- Configurable matplotlib style support

#### 3. Optimized Observing Sequence (`optimize_observing_sequence()`)
Added slew-minimized observing scheduler in `core/magellan_planning.py`:
- Greedy nearest-neighbor algorithm weighted by merit and slew distance
- Respects visibility windows (optimal observation times)
- Generates scheduled UT times for each target
- Outputs total slew distance and cumulative observing time
- Sky map visualization with color-coded observation order

#### 4. Broker Client Standardization
- `FinkLSSTClient` now inherits from `BaseBrokerClient`
- Added `query_alerts()` and `get_stamps()` abstract method implementations
- Exported `AtlasClient` and `Alert` dataclass from `broker_clients`

### Architecture Improvements
- **Reduced run_tonight.py coupling** — report generation now delegated to `ReportGenerator`
- **Consistent type hints** across broker clients
- **Centralized constants** — no more magic numbers in function bodies
- **LaTeX-compatible output** — proper subscript notation in tables and figures

### New Files
```
config.py               — Centralized configuration dataclasses
core/report.py          — PDF report generation module
utils/rubin.mplstyle    — Custom matplotlib style for LaTeX rendering
```

### Updated Code Structure
```
run_tonight.py          — Main CLI (reduced from 1735 to ~1600 lines)
core/
  report.py             — NEW: ReportGenerator class
  magellan_planning.py  — Added optimize_observing_sequence()
  __init__.py           — Export ReportGenerator
broker_clients/
  __init__.py           — Export AtlasClient, Alert
  fink_client.py        — Now inherits BaseBrokerClient
config.py               — NEW: Centralized configuration
```

### Example: Optimized Observing Sequence
For 15 targets on 2026-03-12:
- Total slew: 91° (vs ~400°+ if sorted by merit alone)
- Clusters observations by DDF field to minimize telescope movement
- Color-coded sky map shows observation order (blue→yellow)

---

## Issues Resolved

### Merit Breakdown in Reports
- Added merit component columns (W_time, W_mag, W_prob, W_host, W_ext, W_broker)
- Merit breakdown table page in PDF reports
- Merit function reference page explaining all parameters

### ANTARES P(Ia) Enrichment
- Fink cross-match enrichment for ANTARES-only candidates
- 23/39 candidates got real ML classifier scores
- 10 false positives removed after enrichment

### Report Quality
- Fixed text overlapping in tables
- Consistent merit-based sorting throughout all pages
- LaTeX formatting for scientific notation

---

## Priority Queue (Updated)

### COMPLETED
- ✓ Merit breakdown in reports
- ✓ Fink enrichment for ANTARES candidates
- ✓ Optimized observing sequence (slew-minimized)
- ✓ Code architecture refactor (config.py, report.py)
- ✓ LaTeX-formatted output

### REMAINING
1. **Historical validation** — test on archived DP1 data
2. **Unit tests** — add pytest coverage for core modules
3. **RSP integration** — direct DiaObject photometry queries
