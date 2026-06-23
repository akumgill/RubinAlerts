# RubinAlerts System Architecture

## Overview

RubinAlerts is an automated SN Ia candidate identification and spectroscopic follow-up system for Rubin LSST Deep Drilling Fields. It consists of two subsystems:

1. **Alert Pipeline** — Aggregates transient alerts from multiple brokers, fits light curves, scores candidates for spectroscopic follow-up merit, and generates Magellan observing plans.
2. **LLAMAS Orchestrator** — Converts ranked candidates into executable LLAMAS observing plans with multi-program time accounting for the MAGNETS collaboration.

### MAGNETS Context

MAGNETS is a collaboration of Magellan partner institutions pooling telescope time for Rubin transient follow-up. Each PI brings their TAC-allocated time; the queue scheduler apportions observing time and target prioritization proportionally. The primary instrument is LLAMAS (integral field spectrograph on Magellan/Baade).

---

## System Data Flow

```
   Fink LSST    ANTARES    ALeRCE (ZTF+LSST)
       │            │              │
       └────────────┼──────────────┘
                    │
                    v
        ┌───────────────────────┐
        │   Alert Aggregation   │  Coordinate match, deduplicate,
        │   & Variable Screen   │  reject known variables (13.7k)
        └───────────┬───────────┘
                    │
                    v
        ┌───────────────────────┐
        │   Photometry Fetch    │  Rubin (Fink), ZTF (ALeRCE DB),
        │   & Light Curve Fit   │  ATLAS forced phot (enrichment)
        └───────────┬───────────┘
                    │
                    v
        ┌───────────────────────┐
        │   Host Galaxy &       │  SDSS/PS1/SkyMapper morphology,
        │   Environment         │  NED redshift, nuclear offset,
        │                       │  galactic extinction (SFD)
        └───────────┬───────────┘
                    │
                    v
        ┌───────────────────────┐
        │   Merit Scoring       │  W_time × W_mag × W_prob ×
        │                       │  W_host × W_ext × W_broker
        └───────────┬───────────┘
                    │
                    v
            candidates.csv ──────────────────┐
                    │                         │
                    v                         v
        ┌─────────────────┐      ┌────────────────────────┐
        │  Alert Pipeline │      │   LLAMAS Orchestrator   │
        │  Outputs        │      │                        │
        │  - PDF report   │      │  allocations.yaml      │
        │  - magellan.cat │      │       │                │
        │  - schedule.txt │      │       v                │
        │  - sky maps     │      │  Composite Prioritizer │
        └─────────────────┘      │  (science × budget ×   │
                                 │   phase × keywords)    │
                                 │       │                │
                                 │       v                │
                                 │  Greedy Scheduler      │
                                 │  + Time Accounting     │
                                 │       │                │
                                 │       v                │
                                 │  - timeline.txt        │
                                 │  - TCS catalog         │
                                 │  - summary.txt         │
                                 │  - time_accounting.json│
                                 │  - season_report.txt   │
                                 └────────────────────────┘
```

---

## Alert Pipeline

### Entry Point

```bash
python run_tonight.py <MJD> [--min-prob 0.3] [--days-back 30] [--no-tns]
```

Output: `nights/ut{YYYYMMDD}/` directory with all artifacts.

### Broker Clients

| Broker | Client | Role | API |
|--------|--------|------|-----|
| Fink LSST | `fink_client.py` | Primary Rubin alert stream | REST API |
| ANTARES | `antares_client.py` | Independent Rubin/ZTF broker | Python library |
| ALeRCE ZTF | `alerce_client.py` | ZTF ML classification | REST + PostgreSQL |
| ALeRCE LSST | `alerce_client.py` | Rubin stamp classification | REST API |
| ATLAS | `atlas_client.py` | Forced photometry enrichment | REST batch API |
| TNS | `tns_client.py` | Cross-match, IAU designations | REST API |
| Rubin TAP | `rubin_tap_client.py` | Authoritative LSST photometry | pyvo TAP |

All clients inherit from `BaseBrokerClient` (defined in `base_client.py`) with a common `Alert` dataclass schema.

### Alert Aggregation (`core/alert_aggregator.py`)

- Coordinate matching at 1-2 arcsec tolerance across brokers
- Deduplication with multi-broker agreement bonus
- ANTARES probability harmonization: heuristic P(Ia) proxy capped at 0.50 to prevent single-broker bias

### Light Curve Fitting (`core/peak_fitting.py`)

Dual-fit strategy for robustness:
1. **Inverted parabola** — always converges, per-band
2. **Villar SPM** — multi-band template, preferred when converged

Extracts: peak magnitude, peak time, rise/fall timescales, light curve phase weight.

### Merit Function (`core/magellan_planning.py`)

Multiplicative structure ensures candidates must score well on all factors:

```
Merit = W_time × W_mag × W_prob × W_host × W_ext × W_broker

W_time   = exp(-dt²/2τ²)         τ=10 days, Gaussian decay from peak
W_mag    = Gaussian(m_opt=20.5)   Optimal for Magellan spectroscopy
W_prob   = P(Ia) ∈ [0.1, 1.0]    ML classifier probability
W_host   = {1.0, 0.6, 0.7}       Elliptical, Spiral, Unknown
W_ext    = exp(-E(B-V)/0.15)      Galactic extinction penalty
W_broker = 1.0 + 0.1×(N-1)       Multi-broker agreement bonus
W_moon   = moon/brightness penalty (folded into ranking merit; see below)
```

**Moon penalty in ranking merit** — the lunar-illumination/separation penalty
(`W_moon`) is now multiplied into the merit used for ranking and the within-night
P1–P4 mapping, not applied only as a downstream observability cut. A target that
is faint relative to the prevailing moon phase is therefore ranked lower up front,
so the quartile labels reflect what is actually observable tonight.

### Caching (`cache/alert_cache.py`)

SQLite-backed cache (alerts_cache.db) stores:
- Broker query results (per day, per query params)
- Galaxy info: morphology, redshift, magnitudes, extinction
- NED cross-match results

Prevents API hammering across re-runs and interrupted sessions.

---

## LLAMAS Orchestrator

### Entry Points

```bash
# Basic plan from CSV
python -m orchestrator plan --date 2026-10-15 --targets targets.csv --moon grey

# Full nightly run with time accounting
python -m orchestrator run-nightly --date 2026-10-15 \
    --candidates candidates.csv --allocations allocations.yaml --moon grey

# Post-night reconciliation
python -m orchestrator reconcile --allocations allocations.yaml \
    --program MAGNETS-Stubbs --actual-hours 3.5 --moon grey --date 2026-10-15
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | LLAMASConfig: LCO site (-29.01°, -70.69°, 2380m), 1-min IFU overhead, airmass limit 1.6, exposure table from proposal |
| `models.py` | Target, ScheduledEntry, ProgramAllocation, ObsPlan dataclasses |
| `normalize.py` | Load CSV/RubinAlerts targets, parse coordinates, estimate exposures, merit→P1–P4 mapping, manual-target columns |
| `planner.py` | Twilight calculation, observability windows, greedy scheduling, standard star selection |
| `prioritizer.py` | Composite scoring: science × budget × phase × observability + keywords |
| `accounting.py` | TimeAccountant: per-program budgets (D/G/B), charge/reconcile, JSON persistence |
| `output.py` | Timeline, Magellan TCS catalog, human-readable summary writers |
| `reporting.py` | Nightly time reports, season progress with burn rate projections |
| `run_nightly.py` | End-to-end orchestration tying all modules together |

### Exposure Estimation

Three-tier cascade from Stubbs 2026B proposal Table 1:

| Strategy | Input | Example |
|----------|-------|---------|
| Redshift table | z known | z=0.25 → 95 min (grey) |
| Magnitude scaling | mag known | mag 20 → 45 min, 2.5× per mag |
| Fallback | neither | 45 min default |

**Sub-exposure splitting** — a long integration is broken into N sub-exposures,
each no longer than `LLAMASConfig.max_single_exposure_sec` (default 900 s / 15 min)
to limit cosmic-ray accumulation per frame; the per-frame time is rounded to a
multiple of `LLAMASConfig.exposure_round_sec` (default 10 s) for clean exposure
strings (e.g. `3x600s`). Both thresholds are config-driven, not hardcoded.

### Priority Mapping (merit → P1–P4)

`load_from_rubinalerts` maps the merit score to a P1–P4 label using **within-night
relative quartiles**: top quartile → P1, then P2, P3, bottom → P4. These are
*relative to tonight's candidate set*, not absolute science classes — a P1 on a
sparse night may be weaker than a P4 on a rich one (the summary header repeats this
caveat). When fewer than 4 targets are present the quartile bins degenerate, so the
labels fall back to sorted rank (best → P1, etc.) to avoid collapsing every target
into one tier.

### Scheduling Algorithm

Greedy scorer adapted from Alex's LDSS3 script:

```
score = composite_priority_score - airmass × 10
```

Where `composite_priority_score` combines:
- **Science weight**: P1→100, P2→70, P3→40, P4→20
- **Budget factor**: 1.0 (>5h left), 0.5 (0-5h), 0.1 (exhausted)
- **Phase weight**: w_time from alert pipeline (exp(-dt²/2τ²))
- **Observability**: window_hours / night_hours (0-1)
- **Keywords**: "HIGH PRIORITY" +15, "backup" -15

Falls back to `(5 - priority) × 100 - airmass × 10` when no prioritizer is active (backward compat with `plan` command).

### Time Accounting

Multi-program budgets tracked per moon phase:

```yaml
# allocations.yaml
semester: "2026B"
default_program: "MAGNETS-Stubbs"
programs:
  - program: "MAGNETS-Stubbs"
    pi: "Stubbs"
    allocated_hours: {dark: 5.0, grey: 20.0, bright: 5.0}
  - program: "MAGNETS-Villar"
    pi: "Villar"
    allocated_hours: {dark: 4.0, grey: 8.0, bright: 4.0}
```

**Charge-on-schedule**: Time debited when plan is generated. Post-night reconciliation adjusts for weather losses or target changes. Persistent JSON state with full charge log for audit.

### Manual-Target Columns (CSV input)

`load_targets_csv` accepts manually-curated targets (PIs enqueuing objects not
sourced from alerts). Beyond the required `name, ra, dec`, three optional columns
are honored:

| Column | Effect |
|--------|--------|
| `program` | Charges the target to that program for per-PI accounting. If absent/blank, falls back to the default program **and logs a warning** (never silent mis-attribution). |
| `phase_weight` | Used directly as the target's `w_time`. |
| `peak_mjd` | When `phase_weight` is absent, converted to a phase weight via the same Gaussian as the alert pipeline (`exp(-dt²/2τ²)`, τ=10 d) using the night date. Requires the night MJD; if unavailable, phase stays neutral and a warning is logged. |

If neither `phase_weight` nor `peak_mjd` is given, phase stays neutral (NaN →
treated as 1.0 by `_phase_factor`); a near-peak weight is never fabricated. Missing
optional columns never crash; extra columns are ignored.

---

## External Dependencies

| Service | Purpose | Auth |
|---------|---------|------|
| Fink LSST | Rubin alert stream | None |
| ANTARES | Transient broker | `antares-client` library |
| ALeRCE | ML classification (ZTF+LSST) | None (REST), psycopg2 (DB) |
| ATLAS | Forced photometry | `~/.atlas_credentials` |
| TNS | IAU cross-match | `~/.tns_credentials` |
| Rubin Science Platform | Authoritative LSST photometry | `~/.rsp_token` |
| IRSA | Galactic extinction (SFD) | None |
| NED | Host galaxy redshifts | None |

---

## Project Structure

```
run_tonight.py                    Alert pipeline CLI
supernova_monitor.py              Multi-broker orchestrator class
config.py                         Centralized pipeline configuration

core/
  alert_aggregator.py             Merge/dedup across brokers
  peak_fitting.py                 Light curve fitting (parabola, Villar, SALT)
  magellan_planning.py            Merit scoring, observability, TCS catalog
  variable_screen.py              Known variable rejection (13.7k sources)
  report.py                       PDF report generation
  ddf_fields.py                   LSST DDF field definitions

broker_clients/
  base_client.py                  Abstract broker interface + Alert dataclass
  fink_client.py                  Fink LSST API
  antares_client.py               ANTARES cone search
  alerce_client.py                ALeRCE REST API (ZTF + LSST)
  alerce_db_client.py             ALeRCE direct PostgreSQL
  atlas_client.py                 ATLAS forced photometry (retry/backoff)
  tns_client.py                   TNS cross-match
  rubin_tap_client.py             Rubin Science Platform TAP

orchestrator/
  config.py                       LLAMAS instrument config
  models.py                       Target, ScheduledEntry, ProgramAllocation, ObsPlan
  normalize.py                    Input loading, exposure estimation
  planner.py                      Twilight, observability, greedy scheduler
  prioritizer.py                  Composite priority scoring
  accounting.py                   Multi-program time tracking
  output.py                       Timeline, catalog, summary writers
  reporting.py                    Time reports, season progress
  run_nightly.py                  End-to-end nightly orchestration
  cli.py                          CLI (plan, run-nightly, reconcile)

host_galaxy/
  morphology_filter.py            Host galaxy classification, nuclear offset

utils/
  extinction.py                   Galactic extinction (IRSA SFD)
  ned_query.py                    NED redshift lookups
  coordinates.py                  Angular separation utilities
  catalog_query.py                SDSS/PS1 galaxy queries
  plotting.py                     Light curve visualization

cache/
  alert_cache.py                  SQLite caching (brokers, galaxies, extinction)
```

---

## Key Design Decisions

1. **Multi-broker redundancy** — Four independent alert sources provide resilience to API outages and cross-validation of candidates.

2. **Multiplicative merit** — Product of factors ensures a candidate must score well on ALL dimensions (timing, magnitude, probability, host, extinction, broker agreement). A single high score can't dominate.

3. **Orchestrator separation** — Alert pipeline is instrument-agnostic. LLAMAS scheduling is isolated in `orchestrator/`, enabling future support for other instruments without modifying the alert pipeline.

4. **Greedy scheduling** — Fast (O(N log N)), near-optimal for single-night scheduling. Composite priority score incorporates budget awareness and light curve phase.

5. **Charge-on-schedule with reconciliation** — Simple, prevents over-scheduling. Post-night reconciliation handles weather and target changes.

6. **1-minute IFU overhead** — LLAMAS's integral field unit eliminates slit alignment, giving ~10× better overhead than slit spectrographs. Exploited throughout the scheduling algorithm.

7. **Phase-aware prioritization** — Targets near peak brightness (small delta_t) are automatically boosted via the phase_weight from the alert pipeline's Gaussian decay function.
