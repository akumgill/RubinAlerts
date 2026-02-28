# Implementation Notes

## Project Completion Summary

This document outlines the complete implementation of the Type Ia Supernova Monitor for Elliptical Galaxies in Jupyter Lab.

## Architecture Overview

### 1. Broker Client Layer (`broker_clients/`)

**BaseBrokerClient** (`base_client.py`)
- Abstract base class defining unified interface
- Standard data model: `Alert` dataclass
- Methods: `query_alerts()`, `get_light_curve()`, `get_stamps()`
- Converts broker-specific responses to standard DataFrame format

**AntaresClient** (`antares_client.py`)
- Implements ANTARES broker queries
- Uses `antares-client` library
- Features:
  - Query SN Ia candidates with configurable probability threshold
  - Filter by real-bogus score (RB > 0.8)
  - JSON-based caching with daily expiry
  - Light curve retrieval
  - Postage stamp support (ANTARES specialization)

**AlerceClient** (`alerce_client.py`)
- Implements ALeRCE broker queries
- Uses `alerce` Python client
- Features:
  - Query with Random Forest classifier
  - Extracts all classification probabilities (Ia, II, Ib/c, AGN, VS, Stellar)
  - JSON caching strategy
  - Light curve with MJD support
  - Postage stamp retrieval

### 2. Core Pipeline (`core/`)

**AlertAggregator** (`alert_aggregator.py`)
- Merges alerts from multiple brokers
- Deduplication algorithm:
  - Coordinate matching with 2 arcsecond tolerance
  - Groups detections from different brokers
  - Merges duplicate detections into single records
- Classification comparison:
  - Extracts SN Ia probabilities from all brokers
  - Calculates agreement metrics:
    - Mean probability
    - Standard deviation
    - Min/Max values
    - Agreement score (1.0 if std < 0.1, 0.5 if < 0.25, else 0.0)
- High-confidence filtering by:
  - Minimum Type Ia probability
  - Broker agreement score
  - Number of broker detections

### 3. Galaxy Classification (`host_galaxy/`)

**MorphologyFilter** (`morphology_filter.py`)
- Cross-matches alert coordinates with galaxy catalogs
- Catalog queries:
  - Primary: SDSS (via Astroquery)
  - Fallback: Pan-STARRS
- Morphology classification based on:
  - g-r color index (red sequence: > 0.55)
  - r-i color index (early-type: > 0.15)
  - Magnitude measurements
- Classification categories:
  - "elliptical" - Early-type/red sequence galaxies
  - "spiral" - Blue cloud galaxies
  - "uncertain" - Intermediate colors
  - "unknown" - Insufficient data
- Caching of galaxy information to minimize catalog queries

### 4. Caching System (`cache/`)

**AlertCache** (`alert_cache.py`)
- SQLite database backend for persistent storage
- Tables:
  1. `alerts` - Raw broker alert data
  2. `galaxy_info` - Galaxy morphology and properties
  3. `merged_alerts` - Deduplicated and merged results
- Features:
  - Automatic timestamp tracking
  - Configurable expiry (24 hours default for alerts, 7 days for galaxy info)
  - Query by broker or globally
  - Cache cleanup utilities
  - Spatial queries for galaxy matching (simple degree-based tolerance)

### 5. Utilities (`utils/`)

**CoordinateUtils** (`coordinates.py`)
- Angular separation calculation using Astropy
- Coordinate matching with tolerance
- RA/Dec string parsing and conversion

**CatalogQuery** (`catalog_query.py`)
- SDSS query wrapper with morphology information extraction
- Pan-STARRS query wrapper with color magnitudes
- Morphology classification algorithm:
  - Red sequence identification (early-type galaxies)
  - Color-based classification
  - Error handling for incomplete data

**PlottingUtils** (`plotting.py`)
- Light curve data preparation (organize by band)
- Plotly interactive plotting:
  - Multi-filter support with band-specific colors
  - Error bar visualization
  - Hover tooltips
  - Inverted magnitude axis (brightness convention)
- Matplotlib fallback for display consistency
- Classification comparison charting
- Band color mapping (g→blue, r→orange, i→red, z→purple)

### 6. Main Pipeline (`supernova_monitor.py`)

**SupernovaMonitor** class
- Orchestrates entire workflow
- Methods:
  - `run_full_pipeline()` - Complete query, merge, filter pipeline
  - `query_all_brokers()` - Multi-broker querying with cache support
  - `get_light_curve()` - Retrieve specific light curves
  - `get_stamps()` - Retrieve postage stamps
  - `get_cached_results()` - Access cached previous runs

Workflow:
1. Initialize broker clients
2. Query brokers with caching
3. Merge and deduplicate alerts
4. Filter for Type Ia with confidence thresholds
5. Cross-match with galaxy catalogs
6. Filter for elliptical hosts
7. Calculate agreement metrics
8. Return final candidate list

### 7. Jupyter Notebook (`notebooks/supernova_monitor.ipynb`)

**Interactive Interface Components:**

1. **Setup Cell**
   - Path configuration
   - Imports and logging setup

2. **Monitor Initialization**
   - Creates SupernovaMonitor instance
   - Verifies broker connectivity
   - Confirms cache directory setup

3. **Query and Filter**
   - Configurable parameters (probability, days, cache)
   - Runs full pipeline
   - Reports results

4. **Alert Summary Table**
   - Displays all candidates
   - Key columns: RA/Dec, discovery date, broker agreement
   - Styled DataFrames with formatting

5. **Candidate Selector**
   - Dropdown widget for interactive selection
   - Shows key metrics for each candidate

6. **Classification Comparison**
   - Tabular display of probabilities
   - Agreement metrics and statistics
   - Visual quality indicators

7. **Light Curve Viewer**
   - Retrieves and displays multi-filter light curves
   - Interactive Plotly visualization
   - Photometry table display

8. **Postage Stamps**
   - Retrieves stamp data from broker
   - Placeholder for FITS image display
   - Coordinate information

9. **Export**
   - CSV export with timestamp
   - Summary statistics
   - Broker detection distribution

## Data Flow

```
┌─────────────────────────┐
│   Alert Brokers         │
│ (ANTARES, ALeRCE)       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Broker Clients         │
│  (query_alerts)         │
└────────────┬────────────┘
             │ (caching)
             ▼
┌─────────────────────────┐
│  Alert Cache (SQLite)   │
│  (raw broker data)      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Alert Aggregator       │
│  (merge & deduplicate)  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Galaxy Classifier      │
│  (morphology check)     │
└────────────┬────────────┘
             │ (caching)
             ▼
┌─────────────────────────┐
│  Galaxy Info Cache      │
│  (SDSS/Pan-STARRS)      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Final Candidates       │
│  (Type Ia + elliptical) │
└─────────────────────────┘
```

## Classification Comparison Method

When multiple brokers detect the same object:

1. **Coordinate Matching**: Objects within 2 arcsec are considered the same
2. **Probability Extraction**: Extract SN Ia probability from each broker
3. **Statistics Calculation**:
   - Mean: Average of probabilities
   - Std Dev: Measure of disagreement
   - Min/Max: Range of probabilities
4. **Agreement Score**:
   - Std < 0.1 → Score = 1.0 (high agreement)
   - Std < 0.25 → Score = 0.5 (medium agreement)
   - Std ≥ 0.25 → Score = 0.0 (low agreement)

## Performance Considerations

1. **Caching Strategy**
   - Daily cache expiry for alerts (frequent updates)
   - 7-day expiry for galaxy info (stable data)
   - Significantly reduces API calls on subsequent runs

2. **Catalog Queries**
   - SDSS preferred (more complete for bright objects)
   - Pan-STARRS as fallback
   - Cached to avoid repeated queries

3. **Coordinate Matching**
   - 2 arcsecond tolerance (typical astrometric precision)
   - Vectorized NumPy operations for efficiency
   - Simple degree-based tolerance for simplicity

## Extension Points

The architecture supports easy additions:

1. **New Brokers**: Inherit from `BaseBrokerClient`, implement abstract methods
2. **Additional Catalogs**: Add to `CatalogQuery.query_*()` methods
3. **Classification Features**: Extend `AlertAggregator._add_classification_columns()`
4. **Real-time Streaming**: Could adapt broker clients for Kafka/Pub-Sub
5. **Advanced ML**: Custom classification in merger phase

## Testing Recommendations

1. **Broker Connectivity**
   ```python
   client = AntaresClient()
   results = client.query_alerts(limit=1)
   assert len(results) > 0
   ```

2. **Deduplication**
   - Create test DataFrames with intentional duplicates
   - Verify merging reduces count appropriately

3. **Galaxy Classification**
   - Known elliptical and spiral galaxies
   - Verify correct morphology assignment

4. **Cache**
   - Clear cache, run pipeline, verify cache creation
   - Run again, verify cache usage
   - Check timestamp updates

5. **End-to-End**
   - Run full notebook without errors
   - Verify results are reasonable
   - Check CSV export

## Known Limitations

1. **Broker API Limits**: May require rate limiting for high-frequency queries
2. **Galaxy Data**: Some alerts near survey boundaries may lack morphology classification
3. **Light Curves**: Data availability depends on broker update schedules
4. **Coordinate Matching**: 2 arcsec tolerance may miss close neighbors or misidentify at survey edges
5. **Color-based Morphology**: Simple algorithm; specialized models could improve accuracy
6. **Manual Notebooks**: Could benefit from automation/scheduling

## Future Enhancements

1. Real-time alert streaming (Kafka)
2. Additional brokers (Fink, Lasair, Pitt-Google)
3. Spectroscopic follow-up recommendation engine
4. Machine learning morphology classifier
5. Web dashboard interface
6. Email/Slack notifications
7. Automated candidate vetting
8. Publication-ready analysis tools

## Dependencies

Core dependencies:
- `astropy` - Astronomical calculations and coordinates
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `plotly` - Interactive visualization
- `ipywidgets` - Interactive notebook controls
- `astroquery` - Catalog queries (SDSS, Pan-STARRS)
- `antares-client` - ANTARES broker
- `alerce` - ALeRCE broker
- `sqlalchemy` - (optional) Advanced database features

## File Statistics

- Python modules: 10 files (~1300 lines)
- Jupyter notebook: 1 file (~400 cells)
- Configuration: requirements.txt, README
- Documentation: README.md, IMPLEMENTATION_NOTES.md

## Verification Checklist

- ✓ Project structure created
- ✓ All broker clients implemented
- ✓ Alert aggregation with deduplication
- ✓ Galaxy morphology classification
- ✓ SQLite caching system
- ✓ Coordinate and catalog utilities
- ✓ Plotting utilities for visualization
- ✓ Main monitoring pipeline
- ✓ Interactive Jupyter notebook
- ✓ Comprehensive documentation

## Next Steps for User

1. Install dependencies: `pip install -r requirements.txt`
2. Start Jupyter: `jupyter lab`
3. Open `notebooks/supernova_monitor.ipynb`
4. Run cells sequentially
5. Adjust parameters as needed
6. Compare classifications and examine candidates
7. Export results for analysis

Enjoy monitoring Type Ia supernovae!
