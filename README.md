# RubinAlerts — SN Ia Discovery and Magellan Follow-Up Pipeline

Automated pipeline for discovering Type Ia supernovae in the Rubin Observatory LSST Deep Drilling Fields and generating prioritized spectroscopic follow-up plans for Magellan.

## How to Run

### Nightly pipeline (command line)

```bash
# Generate tonight's observing plan for MJD 61100
python run_tonight.py 61100

# With options
python run_tonight.py 61100 --min-prob 0.3 --max-candidates 200

# Skip ZTF or ATLAS supplementary photometry
python run_tonight.py 61100 --no-ztf --no-atlas

# Skip observability filtering (useful for testing)
python run_tonight.py 61100 --no-observability
```

This creates a night directory `nights/ut20260301/` containing:

| File | Contents |
|------|----------|
| `candidates.csv` | Summary table of all candidates with peak fits and merit scores |
| `magellan_plan.cat` | Magellan TCS catalog (RA-ordered, 16-field format) |
| `observing_schedule.txt` | Human-readable schedule with coordinates, magnitudes, merit |
| `report.pdf` | Multi-page PDF: summary table, diagnostic plots, light curves |
| `lightcurves/*.png` | Per-candidate magnitude-space light curve plots |

### Interactive supervision (notebook)

```bash
jupyter lab notebooks/nightly_supervision.ipynb
```

Review candidates, inspect light curves, edit the observing plan interactively.

### Full multi-broker pipeline (notebook)

```bash
jupyter lab notebooks/supernova_monitor.ipynb
```

Runs the complete ANTARES + ALeRCE + Fink pipeline with classification comparison, variable star screening, and human-in-the-loop classification.

## What the Pipeline Does

1. **Queries Fink** for SN candidates in the Rubin LSST alert stream (`sn_near_galaxy_candidate` and `extragalactic_new_candidate` tags)
2. **Fetches supplementary photometry** from ZTF (via ALeRCE) and ATLAS forced photometry by position match
3. **Combines all photometry** into unified multi-survey light curves (Rubin + ZTF + ATLAS) in nanoJansky flux space
4. **Fits light curves** using both inverted parabola (per-band) and multi-band Villar SPM model with shared explosion epoch
5. **Computes merit scores** based on time since peak and peak brightness
6. **Filters for observability** from Las Campanas (airmass, twilight, hours up)
7. **Generates Magellan plan** sorted by RA (assuming 30 min per observation)

## Installation

```bash
pip install -r requirements.txt
```

### ATLAS credentials

Register at https://fallingstar-data.com/forcedphot/ and create `~/.atlas_credentials`:

```ini
[atlas]
username = your_username
password = your_password
```

### Dependencies

- `astropy`, `scipy`, `numpy`, `pandas`, `matplotlib` — core scientific stack
- `alerce` — ALeRCE broker client (ZTF + LSST classifications)
- `astroquery` — IRSA dust maps, NED redshifts
- `requests` — Fink API, ATLAS API
- `psycopg2-binary` — ALeRCE direct database access (optional, faster bulk queries)
- `pyvo` — ALeRCE TAP queries (optional)
- `sncosmo` — SALT2/SALT3 template fitting (optional)

## Project Structure

```
RubinAlerts/
├── run_tonight.py                 # CLI: nightly pipeline → Magellan plan
├── supernova_monitor.py           # Full multi-broker orchestrator
│
├── broker_clients/
│   ├── fink_client.py             # Fink LSST API (Rubin photometry)
│   ├── alerce_client.py           # ALeRCE (ZTF + LSST classifications)
│   ├── alerce_db_client.py        # ALeRCE direct PostgreSQL (bulk queries)
│   ├── atlas_client.py            # ATLAS forced photometry (batch API)
│   └── antares_client.py          # ANTARES broker
│
├── core/
│   ├── peak_fitting.py            # Parabola + Villar SPM fitting, plots
│   ├── magellan_planning.py       # Merit scores, observability, TCS catalog
│   ├── alert_aggregator.py        # Cross-match, dedup, merge brokers
│   └── ddf_fields.py              # 7 DDF field definitions
│
├── utils/
│   ├── extinction.py              # Galactic dust (IRSA SFD maps)
│   ├── ned_query.py               # NED spectroscopic redshifts
│   ├── coordinates.py             # Coordinate utilities
│   ├── catalog_query.py           # SDSS/PS1 host galaxy queries
│   └── plotting.py                # Light curve visualization
│
├── cache/
│   └── alert_cache.py             # SQLite cache (alerts, extinction, NED)
│
├── notebooks/
│   ├── nightly_supervision.ipynb  # Review run_tonight.py outputs
│   ├── supernova_monitor.ipynb    # Full interactive pipeline
│   └── magellan_planning.ipynb    # Observability and scheduling
│
├── docs/
│   └── rubinalerts_pipeline.tex   # Technical paper (LaTeX)
│
├── requirements.txt
└── nights/                        # Output directories (one per night)
    └── ut20260301/
        ├── candidates.csv
        ├── magellan_plan.cat
        ├── observing_schedule.txt
        ├── report.pdf
        └── lightcurves/
```

## Target Fields

The pipeline searches 7 fields (6 Rubin DDFs + M49):

| Field | RA | Dec | ZTF? | Notes |
|-------|-----|------|------|-------|
| COSMOS | 150.1 | +2.2 | Yes | Northern, full ZTF coverage |
| XMM-LSS | 35.6 | -4.8 | Yes | Northern |
| ECDFS | 53.0 | -28.1 | Marginal | Near ZTF limit |
| ELAIS-S1 | 9.5 | -44.0 | No | Southern, Rubin-only |
| EDFS_a | 58.9 | -49.3 | No | Southern, Rubin-only |
| EDFS_b | 63.6 | -47.6 | No | Southern, Rubin-only |
| M49 | 187.4 | +8.0 | Yes | Virgo Cluster |

## Light Curve Fitting

Two methods run on every candidate:

**Inverted parabola** (per-band): `flux(t) = peak_flux - a*(t-t0)^2`. Fast, always available. Works in nJy flux space.

**Multi-band Villar SPM** (shared explosion epoch): 6-parameter supernova model fit simultaneously across all bands with shared t0 and per-band amplitude/timescale. Uses `scipy.optimize.least_squares` with `soft_l1` robust loss. Better for well-sampled light curves.

The better fit (Villar preferred when converged) provides the headline peak magnitude and time.

## License

Copyright (c) 2025 President and Fellows of Harvard College. All rights reserved.

## Contact

Christopher Stubbs — Harvard University — stubbs@g.harvard.edu
