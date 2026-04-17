# RubinAlerts Lab Notebook

## Project Overview

Automated SN Ia candidate identification pipeline for Rubin LSST Deep Drilling Fields. Aggregates alerts from multiple brokers (Fink, ANTARES, ALeRCE), fits light curves, computes spectroscopic follow-up merit scores, and generates Magellan observing plans.

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

### Remaining — Alert Pipeline

- [ ] Historical validation on archived DP1 data (MJD 60630-60650)
- [ ] Unit tests with pytest coverage
- [ ] Direct RSP DiaObject photometry queries
- [ ] AGN/QSO catalog cross-match (Million Quasar Catalog)

### Remaining — Spectroscopic Orchestration (MAGNETS)

- [ ] Phase 2: Time accounting (track 30hr across D/G/B)
- [ ] Phase 3: RubinAlerts integration (candidates.csv → LLAMAS plan)
- [ ] Phase 4: WD standards (Boyd et al. 2026), reporting

### Known Issues

1. **GLADE+ optical photometry sparse** — most entries have only WISE, morphology returns 'uncertain'
2. **Rubin cadence** — DP1 has sparse early data, limiting light curve quality
3. **TNS cross-match skipped in --fink-only mode** — need to enable in full multi-broker runs
