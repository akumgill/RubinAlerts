# RubinAlerts Lab Notebook

## Project Overview

Automated SN Ia candidate identification pipeline for Rubin LSST Deep Drilling Fields. Aggregates alerts from multiple brokers (Fink, ANTARES, ALeRCE), fits light curves, computes spectroscopic follow-up merit scores, and generates Magellan observing plans.

---

## 2026-03-13 — Data Population Analysis & Fixes

### Session Summary
Diagnosed why merit function weights were defaulting to 1.0. Found critical bugs in E(B-V) propagation and host morphology classification. Added GLADE+ galaxy catalog.

### Data Population Audit

Analyzed 64 final candidates to identify where default values were being used:

| Parameter | Default Rate | Root Cause | Fix Applied |
|-----------|-------------|------------|-------------|
| `w_ext` = 1.0 | 100% | E_BV column never computed | ✓ Fixed |
| `w_host` = 0.7 | 98% | Catalog queries returning NULL | ✓ Fixed |
| `n_ztf` = 0 | 100% | ALeRCE photometry not fetched | Planned |
| `n_atlas` = 0 | 100% | ATLAS forced phot disabled | Planned |

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
- [x] E(B-V) extinction propagation fix
- [x] GLADE+ galaxy catalog integration
- [x] Increased morphology search radius (2 arcmin)

### Remaining
- [ ] Historical validation on archived DP1 data (MJD 60630-60650)
- [ ] Unit tests with pytest coverage
- [ ] Direct RSP DiaObject photometry queries
- [ ] ATLAS forced photometry integration (credentials ready)
- [ ] ZTF photometry via ALeRCE (API configured)
- [ ] NED redshift queries for distance modulus
- [ ] SALT2/SALT3 template fitting (sncosmo ready)

### Known Issues
1. **GLADE+ optical photometry sparse** — most entries have only WISE, morphology returns 'uncertain'
2. **Rubin cadence** — DP1 has sparse early data, limiting light curve quality
3. **ALeRCE ZTF photometry** — API available but not yet integrated into main pipeline
