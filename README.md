# RubinAlerts — SN Ia Discovery and Magellan Follow-Up Pipeline

Automated pipeline for discovering Type Ia supernovae in the Rubin Observatory LSST Deep Drilling Fields and generating prioritized spectroscopic follow-up plans for Magellan.

## How to Run

### Nightly pipeline (command line)

```bash
# Generate tonight's observing plan for MJD 61100
python run_tonight.py 61100

# With options
python run_tonight.py 61100 --min-prob 0.3 --max-candidates 200

# Skip ZTF, ATLAS, or TNS cross-matching
python run_tonight.py 61100 --no-ztf --no-atlas --no-tns

# Query Fink only (faster, skips ANTARES/ALeRCE broker queries)
python run_tonight.py 61100 --fink-only

# Skip observability filtering (useful for testing)
python run_tonight.py 61100 --no-observability
```

This creates a night directory `nights/ut20260301/` containing:

| File                     | Contents                                                                      |
| ------------------------ | ----------------------------------------------------------------------------- |
| `candidates.csv`         | Summary table of all candidates with peak fits and merit scores               |
| `magellan_plan.cat`      | Magellan TCS catalog (RA-ordered, 16-field format)                            |
| `observing_schedule.txt` | Human-readable schedule with coordinates, magnitudes, merit                   |
| `report_{ut_stamp}.pdf`  | Multi-page PDF: title page, summary table, discovery space plot, light curves |
| `lightcurves/*.png`      | Per-candidate magnitude-space light curve plots                               |
| `pipeline.log`           | Full DEBUG-level log of the pipeline run (warnings, errors, timing)           |

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

1. **Queries 4 broker sources for SN candidates** — Fink (Rubin LSST alert stream), ALeRCE-ZTF (ZTF ML classifiers), ALeRCE-LSST (LSST ML classifiers), and ANTARES (ZTF + Rubin heuristics). Each broker contributes candidates from its own classification pipeline.
2. **Merges and deduplicates** across brokers by coordinate matching (1 arcsec tolerance) and shared ZTF object IDs. Computes broker agreement scores.
3. **Screens against known variable star catalogs** (~13,750 variables compiled for the 7 DDFs) to reject contamination.
4. **Cross-matches against TNS** (Transient Name Server) to identify already-reported transients and retrieve spectroscopic classifications.
5. **Fetches Rubin photometry from Fink** — for each candidate, retrieves the full multi-band (g/r/i/z) light curve including DiaSource detections and forced photometry (flux in nanoJanskys from Rubin difference imaging).
6. **Fetches supplementary photometry** from ZTF (via ALeRCE, by position match) and ATLAS forced photometry (cyan/orange bands, batch API via fallingstar.com) for candidates brighter than 20th magnitude.
7. **Combines all photometry** into unified multi-survey light curves (Rubin + ZTF + ATLAS) in nanoJansky flux space.
8. **Applies photometric quality cuts** — requires ≥ 5 points with SNR > 5, detections in ≥ 2 bands, and a time baseline ≥ 2 days. This rejects single-epoch events and single-band artifacts.
9. **Fits light curves** using multi-band Villar SPM model (shared explosion epoch, preferred) with inverted parabola fallback. Both must converge in ≥ 2 bands.
10. **Classifies host galaxies** using SDSS/PS1/SkyMapper/GLADE+ catalogs. Computes nuclear offset to flag potential AGN/TDE (< 1" from host center).
11. **Computes merit scores** — weighted product of time from peak, brightness, P(Ia), host morphology, extinction, and broker agreement.
12. **Filters for observability** from Las Campanas (airmass, twilight, hours up).
13. **Generates Magellan plan** with optimized slew sequence.

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

### TNS credentials (optional)

Register at https://www.wis-tns.org/user and create `~/.tns_credentials`:

```ini
[tns]
api_key = your_api_key
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
│   ├── antares_client.py          # ANTARES broker
│   └── tns_client.py              # TNS cross-match (duplicate detection)
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

| Field    | RA    | Dec   | ZTF?     | Notes                       |
| -------- | ----- | ----- | -------- | --------------------------- |
| COSMOS   | 150.1 | +2.2  | Yes      | Northern, full ZTF coverage |
| XMM-LSS  | 35.6  | -4.8  | Yes      | Northern                    |
| ECDFS    | 53.0  | -28.1 | Marginal | Near ZTF limit              |
| ELAIS-S1 | 9.5   | -44.0 | No       | Southern, Rubin-only        |
| EDFS_a   | 58.9  | -49.3 | No       | Southern, Rubin-only        |
| EDFS_b   | 63.6  | -47.6 | No       | Southern, Rubin-only        |
| M49      | 187.4 | +8.0  | Yes      | Virgo Cluster               |

## Light Curve Fitting

Two methods run on every candidate:

**Inverted parabola** (per-band): `flux(t) = peak_flux - a*(t-t0)^2`. Fast, always available. Works in nJy flux space.

**Multi-band Villar SPM** (shared explosion epoch): 6-parameter supernova model fit simultaneously across all bands with shared t0 and per-band amplitude/timescale. Uses `scipy.optimize.least_squares` with `soft_l1` robust loss. Better for well-sampled light curves.

The better fit (Villar preferred when converged) provides the headline peak magnitude and time.

## License

Copyright (c) 2025 President and Fellows of Harvard College. All rights reserved.

## Contact

Christopher Stubbs — Harvard University — stubbs@g.harvard.edu

Akum Gill - Harvard University - akum@g.harvard.edu
