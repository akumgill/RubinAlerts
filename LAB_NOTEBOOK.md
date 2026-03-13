# RubinAlerts Lab Notebook

## Project Overview

Automated SN Ia candidate identification pipeline for Rubin LSST Deep Drilling Fields. Aggregates alerts from multiple brokers (Fink, ANTARES, ALeRCE), fits light curves, computes spectroscopic follow-up merit scores, and generates Magellan observing plans.

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
| Stage | Count |
|-------|-------|
| Fink candidates | 167 |
| ANTARES candidates | 196 |
| ALeRCE-ZTF | 4 |
| After merge/dedup | 334 |
| With Rubin diaObjectIds | 85 |
| Final (successful fits) | 6 |

**Main bottleneck:** 56 candidates rejected for "too few high-SNR points" — Rubin DP1 has sparse early cadence.

### ANTARES Optimizations Added
- Parallel DDF field searches (ThreadPoolExecutor, 3 workers)
- 60-day date pre-filter (skips loci without recent activity)
- Persistent locus cache (`antares_locus_cache.json`)
- Wall-clock time reduced from ~30 min to ~7 min

### ANTARES Performance by Field
| Field | Checked | Accepted | Notes |
|-------|---------|----------|-------|
| XMM-LSS | 2000 | 0 | 95% rejected as old_activity |
| M49 | ~500 | ~2 | ZTF-heavy, old objects |
| COSMOS | 324 | 28 | Good |
| ECDFS | 597 | 28 | Good |
| ELAIS-S1 | 121 | 28 | Southern, mostly Rubin |
| EDFS_a/b | 34 each | 28 each | Southern, mostly Rubin |

---

## Code Architecture

```
run_tonight.py              — Main CLI, orchestrates pipeline
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
  atlas_client.py           — ATLAS forced photometry

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
- ATLAS: `~/.atlas_credentials`
- ALeRCE DB: hardcoded in `alerce_db_client.py`

---

## Status & Next Steps

### Completed
- [x] Merit breakdown in reports
- [x] Fink enrichment for ANTARES candidates
- [x] Optimized observing sequence (slew-minimized)
- [x] Code architecture refactor (config.py, report.py)
- [x] LaTeX-formatted output
- [x] ANTARES parallel search & date pre-filter
- [x] Host morphology catalog queries (SDSS/PS1/SkyMapper)

### Remaining
- [ ] Historical validation on archived DP1 data (MJD 60630-60650)
- [ ] Unit tests with pytest coverage
- [ ] Direct RSP DiaObject photometry queries
- [ ] ATLAS forced photometry integration (credentials ready)

### Known Issues
1. **Host morphology sparse** — many candidates return "unknown" due to catalog coverage
2. **Rubin cadence** — DP1 has sparse early data, limiting light curve quality
