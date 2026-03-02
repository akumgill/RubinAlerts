# Implementation Notes

## Architecture Overview

RubinAlerts is a multi-broker pipeline for Type Ia supernova discovery in the Rubin Observatory LSST Deep Drilling Fields, with automated Magellan spectroscopic follow-up planning.

### Two entry points

1. **`run_tonight.py`** (CLI) — Nightly pipeline: query all 4 broker sources (Fink + ALeRCE-ZTF + ALeRCE-LSST + ANTARES) → merge/dedup → variable screening → fetch Rubin/ZTF/ATLAS photometry → peak fitting → merit scoring → Magellan observing plan. Outputs to `nights/ut{YYYYMMDD}/`. Key CLI flags: `--fink-only` (skip other brokers), `--no-ztf`, `--no-atlas`, `--no-observability`, `--min-prob`, `--max-candidates`.

2. **`supernova_monitor.py`** (library) — Full multi-broker orchestrator used by the interactive notebooks. Queries ANTARES + ALeRCE (ZTF & LSST) + Fink, cross-matches, deduplicates, screens variables, applies extinction corrections and NED redshifts.

## Data Flow

### run_tonight.py (nightly CLI)

```
Candidate Discovery (all brokers):
  Fink LSST API ──────┐
  ALeRCE-ZTF ─────────┤
  ALeRCE-LSST ────────┤── merge & deduplicate (1 arcsec)
  ANTARES ─────────────┘         │
                                 ▼
                    Variable star screening (DDF catalogs)
                    P(Ia) filtering
                                 │
                                 ▼
Photometry (two-pass):
  Pass 1 — Fink (all candidates):
  ├─ get_light_curve(diaObjectId)            → Rubin grizy (nJy)
  ├─ Circuit breaker: 5 consecutive failures → skip remaining
  └─ Identify bright candidates (< 20th mag)
       │
  Pass 2 — Supplementary (bright only):
  ├─ ATLAS: batch forced photometry          → µJy → nJy (mag < 20 cut)
  └─ ALeRCE: cone search → ZTF detections   → mag → nJy
       │
       ▼
  combine_photometry() → unified DataFrame (mjd, flux, flux_err, band, survey)
       │
       ▼
  fit_parabola() + fit_villar_multiband()
       │
       ▼
  compute_merit(delta_t, peak_mag)
  filter_observable_targets(Las Campanas)
       │
       ▼
  Output: candidates.csv, magellan_plan.cat, report.pdf, lightcurves/, pipeline.log
```

### supernova_monitor.py (full pipeline)

```
ANTARES ──┐
ALeRCE ───┤── merge & deduplicate (1 arcsec) ── variable screening
Fink ─────┘         │
                    ▼
           extinction corrections (IRSA SFD)
           NED redshift lookups
           ATLAS photometry enrichment
                    │
                    ▼
           peak fitting (parabola + Villar + optional SALT2)
           merit scoring → Magellan catalog
```

## Broker Clients

| Client | File | Survey | Data Products |
|--------|------|--------|---------------|
| Fink LSST | `fink_client.py` | Rubin | DiaSource detections, forced photometry (nJy), SN classification scores |
| ALeRCE | `alerce_client.py` | ZTF + LSST | ML classifications (stamp_classifier), light curves, PS1 host data |
| ALeRCE DB | `alerce_db_client.py` | ZTF | Direct PostgreSQL for bulk queries, SPM features |
| ATLAS | `atlas_client.py` | ATLAS | Forced photometry in cyan/orange (µJy), batch API |
| ANTARES | `antares_client.py` | ZTF + Rubin | Heuristic SN proxy, postage stamps |

### Photometry formats

All photometry is standardized to a common DataFrame format before fitting:

| Column | Type | Description |
|--------|------|-------------|
| `mjd` | float | Modified Julian Date |
| `flux` | float | Flux in nanoJanskys |
| `flux_err` | float | Flux uncertainty in nJy |
| `band` | str | Filter name (g, r, i, z, ATLAS-c, ATLAS-o) |
| `survey` | str | Source survey (Rubin, ZTF, ATLAS) |
| `source` | str | detection or forced_phot |
| `magnitude` | float | AB mag (NaN for negative flux) |
| `mag_err` | float | Magnitude uncertainty |

**Flux conversions:**
- Rubin (Fink): already in nJy
- ZTF (ALeRCE): `flux_nJy = 10^((31.4 - mag) / 2.5)`
- ATLAS: `flux_nJy = uJy * 1000`

## Peak Fitting (`core/peak_fitting.py`)

### Light curve cleaning

`clean_light_curve()` runs before every fit:
1. **Deduplication**: when detection + forced_phot share the same (MJD, band), keep the detection (better errors)
2. **Sigma clipping**: per (night, band), reject points > 4σ from the weighted mean flux

### Photometric quality cuts

Before fitting, candidates must pass:
1. **SNR cut**: ≥ 5 data points with SNR > 5
2. **Multi-band**: detections in ≥ 2 photometric bands (among high-SNR points)
3. **Time baseline**: ≥ 2 days between first and last observation (rejects single-epoch events)
4. **Fit convergence**: multiband Villar fit or parabola must converge in ≥ 2 bands

These cuts reject single-epoch transients, cosmic ray artifacts, and poorly-sampled objects that cannot be reliably classified as SNe Ia.

### Inverted parabola (per-band)

`fit_parabola()` — fits `flux(t) = peak_flux - a*(t-t0)^2` independently per band using `scipy.optimize.curve_fit`. Requires ≥ 3 points per band. Used as fallback when Villar fails.

### Multi-band Villar SPM

`fit_villar_multiband()` — fits all bands simultaneously with:
- **Shared t0** (explosion epoch) constrained across all bands
- **Per-band parameters**: amplitude A, beta, gamma, tau_rise, tau_fall
- Uses `scipy.optimize.least_squares` with `soft_l1` robust loss
- Parameter layout: `[t0, A_0, beta_0, gamma_0, tau_rise_0, tau_fall_0, A_1, ...]`

The Villar model in flux space:
```
For mjd < t0 + gamma:
  F = A * (1 - beta*(mjd-t0)/(gamma)) / (1 + exp(-(mjd-t0)/tau_rise))

For mjd >= t0 + gamma:
  F = A * (1-beta) * exp(-(mjd-t0-gamma)/tau_fall) / (1 + exp(-(mjd-t0)/tau_rise))
```

### Merit function

`compute_merit(delta_t, peak_mag)` — Gaussian weight in time (peaks at delta_t=0, σ=10 days) times Gaussian weight in magnitude (peaks at mag=21, σ=2 mag). Higher merit = more valuable for spectroscopic follow-up.

## Magellan Planning (`core/magellan_planning.py`)

- `filter_observable_targets()`: computes twilight times and airmass tracks from Las Campanas for a given UT date
- `write_magellan_catalog()`: generates 16-field TCS format, sorted by RA
- Observing schedule assumes 30 minutes per target

## Target Fields (`core/ddf_fields.py`)

7 fields searched with 1.75° radius: COSMOS, XMM-LSS, ECDFS, ELAIS-S1, EDFS_a, EDFS_b, M49.

Southern DDFs (ELAIS-S1, EDFS_a, EDFS_b) have no ZTF coverage (Dec < -30). These rely entirely on Rubin photometry from Fink.

## Caching (`cache/alert_cache.py`)

SQLite database stores:
- Raw broker alerts (24-hour expiry)
- Galaxy morphology info (7-day expiry)
- Extinction values (keyed by RA/Dec with 0.1 arcmin tolerance)
- NED redshift lookups
- Peak fit results for Magellan planning

## ATLAS Batch Photometry

ATLAS forced photometry is only queried for candidates brighter than 20th magnitude (`ATLAS_BRIGHT_MAG_CUT = 20.0`), since ATLAS's detection limit is ~19.5–20 mag. The pipeline uses a batch approach:

1. After Fink photometry is fetched, identify candidates with any Rubin detection brighter than 20th mag
2. Submit all qualifying targets as a single batch to the ATLAS `radeclist` endpoint (up to 100 per batch)
3. Poll all tasks in parallel until complete
4. Download and parse results into standard nJy format

This is much faster than per-candidate queries — a batch of ~20 targets typically completes in under 10 minutes.

## Resilience and Logging

### Pipeline log file

Every run writes a DEBUG-level log to `nights/ut{date}/pipeline.log`. This captures all warnings, errors, and timing information for post-run diagnostics. Console output remains at INFO level.

### Fink API circuit breaker

The Fink photometry loop tracks consecutive failures. After `FINK_MAX_CONSECUTIVE_FAILURES` (default 5) consecutive timeouts or empty responses, the pipeline skips remaining Fink photometry requests and logs an error. This prevents the pipeline from hanging for hours when the Fink API is down (each failed request takes ~60s to timeout, so 5 failures = ~5 minutes before the circuit breaker trips). Candidates that were already fetched successfully are still fit and scored.

## Known Limitations

1. **ALeRCE LSST cone search is broken** — API ignores ra/dec/radius for LSST. Workaround: query globally, filter locally by DDF coordinates.
2. **LSST stamp classifier doesn't sub-type SNe** — reports P(SN) not P(Ia) vs P(II).
3. **ANTARES P(Ia) is a heuristic proxy**, capped at 0.50.
4. **PS1/SDSS host morphology** only works for northern DDFs (Dec > -30).
5. **ZTF coverage** limited to Dec > -30; southern DDFs are Rubin-only.
6. **ATLAS rate limiting** — batch submissions may take 5-30 minutes for many targets.
