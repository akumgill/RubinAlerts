# Spectroscopic Follow-up Orchestration Layer

**Status**: Draft  
**Author**: Akum Gill  
**Date**: 2026-04-14  
**Collaborators**: Chris Stubbs, Ashley Villar's group (Yize Dong), MAGNETS consortium

## Overview

This document describes the design of an orchestration layer that connects target requests to automated observing plan generation for spectroscopic follow-up of transients. The system sits between human-initiated target requests and telescope-specific scheduling scripts, providing automated prioritization and time accounting.

This infrastructure supports the **MAGNETS collaboration** (Magellan partner institutions pooling time for Rubin transient follow-up) and implements the "collaborative infrastructure needed to execute this blended program" (Stubbs 2026B proposal).

## Background

### Current State (Ashley's Group)

Based on meeting with Yize Dong (2026-04-14):

1. **Request intake**: Google Form → Google Sheet
   - Columns: Requester, Instrument, Target name, Description, RA/Dec (degrees), YSE-PZ link, Brightness, Photometry request, Notes, Email, Status
   - No formal priority field

2. **Plan generation**: Manual notebook (Yize)
   - Ad-hoc ranking by visibility and brightness
   - Plots airmass curves, manually orders targets
   - Generates catalog + timeline files

3. **Existing tooling**: Alex's LDSS3 script (`generate_obsplan.py`)
   - Greedy scheduler with priority support (P1-P4)
   - Automatic standard star selection
   - LaTeX observing package generation
   - Not currently used due to input format mismatch

### MAGNETS Collaboration Context

From the Stubbs 2026B proposal (propid 2835):

> "MAGNETS, a collaboration between several Magellan partner institutions, is a brand new effort motivated by the start of LSST alerts in March of this year... Our collaboration's plan is to pool our awarded observing time and develop an internal queue schedule to address a wide range of science goals."

**Key requirements from proposal:**
- Queue-based observational strategy
- Time apportioned in proportion to TAC awards
- Multiple science cases: SNe Ia cosmology (Stubbs), core collapse (Villar), exotic transients (others)
- Student projects declared and protected

**Instrument for this program:**
- **LLAMAS** (Magellan/Baade): Integral field spectrograph for SNe Ia spectrophotometry

*Note: The LDSS3 scripts from Alex/Yize are reference material for how scheduling algorithms work, but this program uses LLAMAS exclusively.*

### Problem Statement

Chris Stubbs: "We're going to submit an independent but coordinated proposal. That all sounds too manual to sustain. Needs to be much more automated including some kind of time accounting."

**Pain points:**
- Manual priority assignment doesn't scale with multiple requesters
- No time budget tracking per program/PI
- Input format mismatch between request sheet and scheduling script
- Multi-instrument requests require separate workflows
- Queue scheduling across diverse science cases

## Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F1 | Ingest target requests from Google Sheet (or equivalent) | Must |
| F2 | Normalize heterogeneous inputs to common schema | Must |
| F3 | Score/rank targets by composite priority | Must |
| F4 | Track time allocations per program/PI | Must |
| F5 | Generate observing plans for LLAMAS (Magellan/Baade) | Must |
| F6 | Handle backup target lists | Should |
| F7 | Integrate with RubinAlerts merit-scored candidates | Should |
| F8 | Support white dwarf spectrophotometric standards | Should |
| F9 | Moon phase-aware scheduling (dark/grey/bright) | Should |

### Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NF1 | Run nightly without manual intervention |
| NF2 | Produce human-readable audit trail |
| NF3 | Graceful degradation if external APIs fail |
| NF4 | Compatible with existing SkyPortal data flow |

## Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          External Systems                               │
├─────────────────┬─────────────────┬─────────────────┬──────────────────┤
│  Google Sheet   │   RubinAlerts   │    YSE-PZ /     │    SkyPortal     │
│  (manual req)   │   (automated)   │    ALeRCE       │   (post-obs)     │
└────────┬────────┴────────┬────────┴────────┬────────┴────────┬─────────┘
         │                 │                 │                 │
         v                 v                 v                 │
┌────────────────────────────────────────────────────┐        │
│            Orchestration Layer                      │        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │        │
│  │   Ingester   │→ │  Prioritizer │→ │  Router   │ │        │
│  └──────────────┘  └──────────────┘  └─────┬─────┘ │        │
│         ↑                ↑                 │       │        │
│  ┌──────────────┐  ┌──────────────┐        │       │        │
│  │  Normalizer  │  │ Time Acctng  │        │       │        │
│  └──────────────┘  └──────────────┘        │       │        │
└────────────────────────────────────────────┼───────┘        │
                                             │                │
         ┌───────────────────────────────────┼────────────────┘
         │                                   │
         v                                   v
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  LDSS3 Planner  │  │ Binospec Planner│  │  FAST Planner   │
│  (Alex script)  │  │   (future)      │  │   (future)      │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         v                    v                    v
┌─────────────────────────────────────────────────────────────┐
│                     Output Artifacts                         │
│  catalog.cat  |  timeline.txt  |  ObsPlan.pdf  |  finders/  │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Ingester

Pulls targets from multiple sources:

```python
class TargetIngester:
    """Pull targets from configured sources."""
    
    def ingest_google_sheet(self, sheet_id: str, range: str) -> list[RawTarget]:
        """Fetch rows from Google Sheet via Sheets API."""
        
    def ingest_rubinalerts(self, candidates_csv: Path) -> list[RawTarget]:
        """Read RubinAlerts merit-scored candidates."""
        
    def ingest_manual(self, targets_file: Path) -> list[RawTarget]:
        """Read tab-separated target file."""
```

**Google Sheet schema** (current):
```
Timestamp | Name of Requester | Instrument Requested | Name of Target | 
Brief description | R.A. (J2000 degrees) | Dec. (J2000 degrees) | 
YSE-PZ link | Current Apparent Brightness and Filter | 
Photometry requested? | Additional notes | Email Address | Status
```

#### 2. Normalizer

Converts heterogeneous inputs to canonical schema:

```python
@dataclass
class Target:
    """Canonical target representation."""
    name: str
    ra_deg: float
    dec_deg: float
    ra_hms: str  # HH:MM:SS.ss
    dec_dms: str  # +DD:MM:SS.s
    
    mag: float
    mag_filter: str  # 'r', 'g', 'i', etc.
    
    instrument: str  # 'LDSS3', 'Binospec', 'FAST'
    priority: int  # 1 (highest) to 4 (lowest)
    
    requester: str
    program: str  # For time accounting
    
    exposure_str: str  # '3x900s'
    n_exp: int
    exp_sec: int
    
    description: str
    notes: str
    url: str  # YSE-PZ, ALeRCE, etc.
    
    # Computed fields
    observability_score: float | None = None
    composite_priority: float | None = None
```

**Normalization rules:**

| Source field | Transformation |
|-------------|----------------|
| RA (degrees) | → `ra_hms` via astropy |
| Dec (degrees) | → `dec_dms` via astropy |
| "19.5, r-band" | → `mag=19.5`, `mag_filter='r'` |
| "3x900s" or "2700s total" | → `n_exp=3`, `exp_sec=900` |
| No priority given | → `priority=2` (default) |

**Exposure time estimation** (if not provided):
```python
def estimate_exposure(mag: float, instrument: str) -> tuple[int, int]:
    """Estimate n_exp, exp_sec from magnitude."""
    # LDSS3 empirical scaling
    if instrument == 'LDSS3':
        if mag < 18:
            return 2, 600
        elif mag < 19:
            return 2, 900
        elif mag < 20:
            return 3, 900
        elif mag < 21:
            return 3, 1200
        else:
            return 4, 1200
```

#### 3. Prioritizer

Computes composite priority score:

```python
def compute_priority(target: Target, 
                     obs_date: str,
                     time_budgets: dict[str, float]) -> float:
    """
    Composite priority = science_weight × observability × time_factor
    
    Components:
    - science_weight: From explicit priority (P1=1.0, P2=0.7, P3=0.4, P4=0.2)
    - observability: Hours above airmass limit / total dark hours
    - time_factor: 1.0 if program has budget, 0.1 if exhausted
    """
```

**Priority sources** (in order):
1. Explicit priority in request ("HIGH PRIORITY", "low prio")
2. RubinAlerts merit score (if from pipeline)
3. Description keywords ("classification needed" → P1, "backup" → P4)
4. Default: P2

#### 4. Time Accounting

Tracks allocations and usage per program:

```python
@dataclass
class TimeAllocation:
    program: str
    pi: str
    semester: str
    allocated_hours: float
    used_hours: float
    
    @property
    def remaining(self) -> float:
        return self.allocated_hours - self.used_hours

class TimeAccountant:
    """Track time budgets per program."""
    
    def __init__(self, allocations_file: Path):
        self.allocations = self._load(allocations_file)
        
    def charge(self, program: str, hours: float) -> bool:
        """Deduct time from program budget. Returns False if insufficient."""
        
    def get_factor(self, program: str) -> float:
        """Priority multiplier based on remaining budget."""
        remaining = self.allocations[program].remaining
        if remaining > 5:
            return 1.0
        elif remaining > 0:
            return 0.5  # Conserve remaining time
        else:
            return 0.1  # Deprioritize but don't exclude
```

**Allocations file** (`allocations.yaml`):
```yaml
semester: 2026B
allocations:
  # Stubbs 2026B: 0.5D + 2.0G + 0.5B = 30 hours
  - program: stubbs-snia
    pi: Chris Stubbs
    instrument: LLAMAS
    allocated_hours: 30
    used_hours: 0.0
    moon_phases:
      dark: 5
      grey: 20
      bright: 5
    
  # Villar program (example - awaiting TAC award)
  - program: villar-ccsn
    pi: Ashley Villar
    instrument: LDSS3
    allocated_hours: 16
    used_hours: 0.0
```

#### 5. LLAMAS Planner

Generates observing plans for LLAMAS integral field spectrograph. Adapts scheduling logic from the LDSS3 reference script but with LLAMAS-specific considerations:

**LLAMAS advantages** (from proposal):
- IFU: no slit losses, immediate integration after slew
- Host galaxy spectrum "for free" within FOV
- 1 minute overhead per observation (vs 10+ for slit spectrographs)

```python
class LLAMASPlanner:
    """Generate LLAMAS observing plans for Magellan/Baade."""
    
    def __init__(self):
        self.observatory = EarthLocation.of_site('Las Campanas Observatory')
        self.standards = self._load_wd_standards()  # Boyd et al. 2026
        
    def generate(self, targets: list[Target], obs_date: str, 
                 moon_phase: str) -> ObsPlan:
        """
        Generate observing plan for a night.
        
        Args:
            targets: Prioritized target list
            obs_date: YYYY-MM-DD
            moon_phase: 'dark', 'grey', or 'bright'
        """
        # 1. Calculate twilight times
        evening, morning = self._calculate_twilight(obs_date)
        
        # 2. Filter targets by moon phase (fainter targets need darker time)
        eligible = self._filter_by_moon(targets, moon_phase)
        
        # 3. Compute observability windows
        for t in eligible:
            t.obs_window = self._find_window(t.coord, evening, morning)
            
        # 4. Schedule with greedy algorithm
        schedule = self._greedy_schedule(eligible, evening, morning)
        
        # 5. Add WD standards at start/end
        schedule = self._add_standards(schedule, evening, morning)
        
        return ObsPlan(schedule=schedule, backups=self._get_backups(eligible, schedule))
        
    def _estimate_exposure(self, mag: float, z: float, moon_phase: str) -> tuple[int, int]:
        """
        Estimate exposure time based on proposal Table 1.
        
        Returns (n_exp, exp_sec) tuple.
        """
        # From Stubbs 2026B proposal
        if z < 0.20:  # r ~ 19.3
            return (1, 35 * 60)  # 35 min
        elif z < 0.30:  # r ~ 20.6
            return (1, 95 * 60)  # 95 min
        elif z < 0.35:  # r ~ 21.3
            if moon_phase == 'dark':
                return (1, 45 * 60)  # 45 min (dark)
            else:
                return (1, 160 * 60)  # 160 min (grey)
        else:  # z ~ 0.35-0.40, r ~ 21.9
            return (1, 180 * 60)  # 180 min (dark only)
```

**Key differences from LDSS3 scheduling:**
- IFU overhead is ~1 min vs 10 min for slit setup
- Spectral binning allows fainter targets in reasonable time
- Moon phase strongly affects faint-target feasibility
- WD standards interleaved (not just start/end)

### Data Flow Example

```
1. Nightly trigger (cron or manual)
   |
2. Ingester pulls from Google Sheet
   → 15 targets (mixed instruments)
   |
3. Normalizer converts to canonical schema
   → 15 Target objects with ra_hms, dec_dms, etc.
   |
4. Prioritizer scores each target
   → Composite priorities computed
   → 3 targets deprioritized (program over budget)
   |
5. Router groups by instrument
   → LDSS3: 8 targets
   → Binospec: 5 targets  
   → FAST: 2 targets
   |
6. LDSS3Planner generates plan
   → Scheduled: 5 targets (night too short for all 8)
   → Backup: 3 targets
   |
7. Time Accountant charges programs
   → stubbs-snia: +3.2 hours
   → villar-precursors: +1.8 hours
   |
8. Outputs written
   → output/ldss3_2026-04-14/ObsPlan/
   → output/ldss3_2026-04-14/LDSS_20260414_Observer.pdf
```

## Interface Specifications

### Input: Google Sheet API

Using `gspread` with service account:

```python
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

def fetch_sheet(sheet_id: str, range_name: str) -> list[dict]:
    creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet(range_name)
    return worksheet.get_all_records()
```

### Input: RubinAlerts Integration

Read from pipeline output:

```python
def ingest_rubinalerts(candidates_csv: Path, min_merit: float = 0.5) -> list[Target]:
    """Convert RubinAlerts candidates to Target objects."""
    df = pd.read_csv(candidates_csv)
    df = df[df['merit_score'] >= min_merit]
    
    targets = []
    for _, row in df.iterrows():
        targets.append(Target(
            name=row['object_id'],
            ra_deg=row['ra'],
            dec_deg=row['dec'],
            mag=row['peak_mag'],
            mag_filter=row['peak_filter'],
            instrument='LDSS3',  # Default for Magellan
            priority=merit_to_priority(row['merit_score']),
            program='stubbs-snia',
            description=f"SN Ia candidate, P(Ia)={row['p_snia']:.2f}",
            url=row.get('tns_url', ''),
            ...
        ))
    return targets
```

### Output: Observing Artifacts

Same structure as Alex's script:

```
output/ldss3_2026-04-14/
├── ObsPlan/
│   ├── Observer_LDSS_0414_catalog    # TCS catalog
│   ├── Observer_LDSS_0414_timeline   # UT schedule
│   └── Finders/                      # Finder charts
├── LDSS_20260414_Observer.tex
├── LDSS_20260414_Observer.pdf
└── time_accounting.json              # NEW: charges for this run
```

## CLI Interface

```bash
# Full nightly run
python -m orchestrator.run_nightly \
    --date 2026-04-14 \
    --sheet-id 1abc...xyz \
    --output-dir output/

# Include RubinAlerts candidates
python -m orchestrator.run_nightly \
    --date 2026-04-14 \
    --sheet-id 1abc...xyz \
    --rubinalerts-csv candidates_20260414.csv \
    --output-dir output/

# Dry run (no time charges)
python -m orchestrator.run_nightly \
    --date 2026-04-14 \
    --sheet-id 1abc...xyz \
    --dry-run

# Single instrument
python -m orchestrator.run_nightly \
    --date 2026-04-14 \
    --sheet-id 1abc...xyz \
    --instrument LDSS3
```

## Configuration

`config/orchestrator.yaml`:

```yaml
# Target sources
google_sheet:
  sheet_id: "1abc...xyz"
  range: "Requests!A:M"
  credentials: "credentials.json"

rubinalerts:
  enabled: true
  min_merit: 0.5
  default_program: stubbs-snia

# LLAMAS instrument configuration
llamas:
  telescope: Magellan Baade
  observatory:
    name: Las Campanas Observatory
    lat: -29.0142
    lon: -70.6925
    elevation: 2380
  overhead_minutes: 1  # IFU advantage
  max_airmass: 1.6

# Target selection (from proposal)
target_selection:
  magnitude_range: [18.0, 21.5]  # r-band
  redshift_range: [0.1, 0.4]
  prefer_elliptical_hosts: true
  require_near_peak: true

# Moon phase scheduling (from proposal Table 1)
moon_scheduling:
  bright:
    max_redshift: 0.20
    typical_exposure_min: 35
  grey:
    max_redshift: 0.35
    typical_exposure_min: 95
  dark:
    max_redshift: 0.40
    typical_exposure_min: 180

# White dwarf standards (Boyd et al. 2026)
standards:
  catalog: config/wd_standards.csv
  observe_per_night: 2-3
  interleave: true

# Time accounting
time_accounting:
  allocations_file: config/allocations.yaml
  charge_on_schedule: true

# Priority keywords
prioritization:
  default_priority: 2
  keywords:
    high_priority: ["HIGH PRIORITY", "classification needed", "rising", "near peak"]
    low_priority: ["backup", "low prio", "if time"]
```

## Implementation Plan

### Phase 1: Core Pipeline (Week 1-2)

1. **Normalizer module**
   - Coordinate conversion (deg → HMS/DMS)
   - Magnitude/filter parsing
   - Redshift-based exposure time estimation (per proposal Table 1)
   
2. **LLAMAS Planner**
   - Greedy scheduling algorithm (adapt from LDSS3 reference)
   - Moon phase-aware target selection
   - 1-minute overhead model
   - WD standard star interleaving

3. **Basic CLI**
   - Read from file input
   - Generate LLAMAS observing plan

### Phase 2: Prioritization & Accounting (Week 3)

4. **Prioritizer module**
   - Science priority from RubinAlerts merit scores
   - Observability calculation
   - Elliptical host preference
   - Near-peak timing bonus

5. **Time Accountant**
   - Track 30-hour allocation (5D + 20G + 5B)
   - Moon phase budget tracking
   - Charge/query interface

### Phase 3: Integration (Week 4)

6. **RubinAlerts integration**
   - Read candidates.csv from nightly pipeline
   - Merit → LLAMAS priority mapping
   - Filter to z=0.1-0.4, r=18-21.5

7. **Google Sheet ingester** (optional)
   - gspread integration
   - Manual target submission support

### Phase 4: Polish (Week 5+)

8. **Reporting**
   - Nightly plan summary
   - Time usage tracking
   - Season progress dashboard

9. **WD Standards handling**
   - Boyd et al. 2026 catalog integration
   - Automatic selection for calibration

## Open Questions

1. **Time accounting granularity**: Charge on schedule or on observation? (Propose: schedule, with reconciliation)

2. **Priority conflicts**: How to handle when multiple programs request same target?

3. **Backup target policy**: Should backups count against time budget?

4. **Multi-night planning**: Should we look ahead to optimize across nights?

5. **Status feedback**: How to mark targets as "observed" in Google Sheet?

## References

### Local Files
- Alex's LDSS3 script: `ref/LDSS_ObsPlan_Generator/generate_obsplan.py`
- Yize's notebook: `ref/march_obs_run/obs_plan.ipynb`
- Google Sheet schema: `ref/march_obs_run/targets_magellan.txt`
- Stubbs 2026B proposal: `ref/cstubbs2026B (2) (1).pdf`
- RubinAlerts pipeline: `run_tonight.py`, `core/magellan_planning.py`

### Key Papers
- Adame et al. 2025: DESI evolving dark energy constraints
- Boyd et al. 2026: White dwarf spectrophotometric standards
- Ivezic et al. 2019: LSST science drivers

### Proposal Details (Stubbs 2026B, propid 2835)
- Semester: 2026B (Jul 7 - Jan 16)
- Instrument: LLAMAS Integral Field Spectrograph
- Telescope: Magellan Baade
- Nights: 0.5D + 2.0G + 0.5B (30 hours total)
- Targets: 29 (SNe Ia in DDFs + WD standards)
- Magnitude range: 18.0 <= r <= 21.5
- Redshift range: 0.1 < z < 0.4
- Queued observing: Yes
