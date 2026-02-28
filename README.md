# Type Ia Supernova Monitor for Elliptical Galaxies

A comprehensive Jupyter Lab notebook system for monitoring Type Ia supernovae in elliptical galaxies using multiple alert brokers.

## Overview

This project monitors **ANTARES** and **ALeRCE** brokers to identify Type Ia supernova candidates in elliptical host galaxies. It provides:

- **Broker Aggregation**: Queries and merges alerts from multiple sources
- **Deduplication**: Cross-matches coordinates to identify the same objects across brokers
- **Classification Comparison**: Compare supernova classifications from different brokers
- **Galaxy Filtering**: Cross-matches with galaxy catalogs to identify elliptical hosts
- **Interactive Visualization**: Examine light curves, postage stamps, and classifications
- **Local Caching**: Minimizes API calls through persistent local storage

## Quick Start

### Installation

```bash
# Clone/navigate to project directory
cd RubinAlerts

# Install dependencies
pip install -r requirements.txt
```

### Using the Notebook

```bash
# Start Jupyter Lab
jupyter lab

# Open notebooks/supernova_monitor.ipynb
```

### Command-Line Usage

```python
from supernova_monitor import SupernovaMonitor

monitor = SupernovaMonitor()

# Run full pipeline
results = monitor.run_full_pipeline(
    min_ia_probability=0.7,
    days_back=30
)

# Query light curve
lc = monitor.get_light_curve(object_id, broker='ANTARES')

# Get postage stamps
stamps = monitor.get_stamps(object_id, ra, dec, broker='ANTARES')
```

## Project Structure

```
RubinAlerts/
├── broker_clients/           # Broker query clients
│   ├── base_client.py       # Abstract interface
│   ├── antares_client.py    # ANTARES broker client
│   └── alerce_client.py     # ALeRCE broker client
│
├── core/                     # Core pipeline
│   └── alert_aggregator.py  # Alert merging and deduplication
│
├── host_galaxy/             # Galaxy classification
│   └── morphology_filter.py # Galaxy morphology filtering
│
├── cache/                   # Caching system
│   ├── alert_cache.py      # SQLite-based cache
│   └── data/               # Cached data files
│
├── utils/                   # Utilities
│   ├── coordinates.py      # Coordinate operations
│   ├── catalog_query.py    # Galaxy catalog queries
│   └── plotting.py         # Visualization utilities
│
├── notebooks/              # Jupyter notebooks
│   └── supernova_monitor.ipynb  # Main interactive notebook
│
├── supernova_monitor.py    # Main monitoring class
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Features

### 1. Multi-Broker Monitoring

Connects to:
- **ANTARES** (Arizona-NOIRLab Temporal Analysis & Response to Events System)
  - Real-time broker with excellent postage stamp support
  - Python client: `antares-client`

- **ALeRCE** (Automatic Learning for the Rapid Classification of Events)
  - Fast ML-based classifications
  - Python client: `alerce`

### 2. Alert Aggregation

- Queries both brokers for Type Ia supernova candidates
- Configurable probability thresholds
- Merges detections from multiple brokers
- Deduplicates using 2 arcsecond coordinate tolerance
- Tracks which brokers detected each object

### 3. Classification Comparison

- Side-by-side comparison of classifications:
  - Type Ia probability
  - Type II probability
  - Type Ib/c probability
  - AGN probability
- Agreement scoring across brokers
- Confidence statistics (mean, std dev, min, max)

### 4. Galaxy Morphology Classification

Cross-matches with galaxy catalogs:
- **SDSS** (Sloan Digital Sky Survey)
- **Pan-STARRS**

Classification based on:
- Color indices (g-r, r-i)
- Concentration index
- Red sequence identification

Filters for elliptical galaxies as Type Ia hosts.

### 5. Interactive Visualization

**Alert Summary Table**
- Object coordinates, discovery date
- Host galaxy type
- Broker agreement metrics
- Sortable/filterable interface

**Classification Comparison**
- Bar charts of probabilities by broker
- Detailed classification tables
- Agreement metrics

**Light Curve Viewer**
- Multi-filter light curves (g, r, i, z)
- Interactive Plotly plots
- Error bars and uncertainty bands
- Phase-folded option

**Postage Stamps**
- Science, reference, difference images
- Multiple epochs
- FITS header information
- ANTARES specialization

**Export**
- CSV export of results
- Summary statistics

## Configuration

Edit parameters in the notebook's "Query and Filter Alerts" cell:

```python
MIN_IA_PROBABILITY = 0.7   # Minimum Type Ia confidence
DAYS_BACK = 30              # Query window (days)
USE_CACHE = True            # Use local caching
```

Filter for high-confidence candidates in "Alert Aggregation":
- Type Ia probability threshold
- Broker agreement requirement
- Minimum number of broker detections

## Caching

Local SQLite database stores:

1. **Raw broker alerts** (24-hour expiry)
   - Alert IDs, coordinates, classifications
   - Photometry metadata

2. **Galaxy information** (7-day expiry)
   - Morphology classifications
   - Cross-match results
   - Redshifts and magnitudes

3. **Merged alert results** (24-hour expiry)
   - Deduplicated alerts
   - Classification comparisons

Clear old cache:
```python
monitor.cache.clear_old_cache(days_old=7)
```

## API Limitations

Be aware of broker API rate limits:
- ANTARES: Check documentation for current limits
- ALeRCE: Typically generous for research use

The caching system minimizes API calls. Most queries use cached data.

## Data Privacy & Attribution

- ANTARES: https://antares.noirlab.edu/
- ALeRCE: https://alerce.science/
- SDSS: https://www.sdss.org/
- Pan-STARRS: https://panstarrs.stsci.edu/

Cite brokers appropriately in publications.

## Troubleshooting

### No candidates found

- Check internet connection (broker queries)
- Verify API credentials if required
- Lower probability thresholds
- Increase `DAYS_BACK` parameter
- Check broker status pages

### Missing light curve/stamps

- Object may not be available in selected broker
- Try alternate broker
- Check alert is recent enough for data availability

### Galaxy classification fails

- Internet needed for catalog queries
- Some objects near survey edges may lack morphology data
- Catalog query limits may cause timeouts

### Cache issues

Clear cache and restart:
```bash
rm -rf cache/data/*.db
```

## Future Enhancements

- Additional brokers (Fink, Lasair, Pitt-Google)
- Real-time alert streaming
- Spectroscopic follow-up suggestions
- Machine learning classification refinement
- Automated email notifications
- Web interface

## Contributing

Improvements and bug reports welcome!

## License

Copyright (c) 2025 President and Fellows of Harvard College. All rights reserved.

This software is provided for academic and research purposes only. No commercial
use, redistribution for commercial purposes, or incorporation into commercial
products is permitted without prior written authorization from the copyright holder.

## Contact

Christopher Stubbs
Harvard University
stubbs@g.harvard.edu

## References

- Brokers: https://antares.noirlab.edu/, https://alerce.science/
- Vera C. Rubin Observatory: https://www.lsst.org/
- LSST Alert System: https://dmtn-102.lsst.io/
