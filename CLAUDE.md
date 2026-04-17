# CLAUDE.md — RubinAlerts

## Project Overview

Automated SN Ia candidate identification pipeline for Rubin LSST Deep Drilling Fields. Two main subsystems:

1. **Alert Pipeline** (`run_tonight.py`): Aggregates alerts from 5 brokers (Fink, ANTARES, ALeRCE, ATLAS, TNS), fits light curves, computes merit scores, generates Magellan observing plans.
2. **LLAMAS Orchestrator** (`orchestrator/`): Spectroscopic scheduling for the MAGNETS collaboration — converts ranked target lists into LLAMAS observing plans with greedy scheduling, exposure estimation, and standard star interleaving.

## Environment

- **Conda env:** `RubinAlerts` (Python 3.12)
- **Activate:** `conda activate RubinAlerts`
- **Key dependencies:** astropy, pandas, numpy, sncosmo, antares-client, psycopg2-binary, requests, matplotlib

## Running

```bash
# Alert pipeline
python run_tonight.py 61101 --min-prob 0.3 --days-back 30

# LLAMAS orchestrator
python -m orchestrator --date 2026-10-15 --targets ref/test_targets.csv --moon grey --output-dir /tmp/test/
```

## Code Conventions

- `@dataclass` for config and data models (not dicts)
- `logging.getLogger(__name__)` in every module
- `float('nan')` for optional numeric fields (check with `math.isfinite()`)
- Astropy `Time`, `SkyCoord`, `u.deg/u.hour` for all astronomical quantities
- Module-level default config instances (e.g., `LLAMAS_CONFIG = LLAMASConfig()`)
- Broker clients follow `BaseBrokerClient` interface with `query_alerts()`, `get_stamps()`

## Credentials

- RSP TAP: `~/.rsp_token`
- ATLAS: `~/.atlas_credentials` (INI format, `[atlas]` section)
- TNS: `~/.tns_credentials`
- ALeRCE DB: hardcoded in `broker_clients/alerce_db_client.py`

## Key Architecture Decisions

- **LLAMAS only** — the orchestrator is for LLAMAS on Magellan/Baade exclusively. LDSS3 materials in `ref/` are reference only.
- **Greedy scheduling** — score = `(5 - priority) * 100 - airmass * 10`, targets scheduled in priority/airmass order within observability windows.
- **Exposure estimation cascade:** redshift table (proposal Table 1) -> magnitude scaling (mag 20 = 45 min, 2.5x per mag) -> fallback (45 min).
- **1-minute IFU overhead** (not 10 min like slit instruments).

## Testing

```bash
# Orchestrator on test targets
python -m orchestrator --date 2026-10-15 --targets ref/test_targets.csv --moon grey --output-dir /tmp/test/ --verbose

# Broker import check
python -c "from broker_clients.atlas_client import AtlasClient; c = AtlasClient(); print(c.verify_credentials())"
```

## Output Locations

- Alert pipeline: `nights/ut{YYYYMMDD}/`
- Orchestrator: `--output-dir` flag, files named `LLAMAS_{date}_{timeline,catalog,summary}.txt`
