# Quick Start Guide

## Installation (5 minutes)

```bash
# Navigate to project directory
cd /Users/christopherstubbs/Desktop/projects/RubinAlerts

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install basic dependencies
pip install -r requirements-basic.txt

# Install ANTARES client (optional, may require git)
# If this fails, you can skip it - the system works with ALeRCE alone
pip install git+https://github.com/astro-umar/antares-client.git
```

For detailed installation help, see **INSTALL.md**

## Running the Notebook (2 minutes)

```bash
# Start Jupyter Lab
jupyter lab

# In browser, navigate to: notebooks/supernova_monitor.ipynb
# Run cells from top to bottom
```

## What You'll Get

The notebook will:

1. **Query Brokers** - Retrieve recent Type Ia supernova candidates from ANTARES and ALeRCE
2. **Merge Results** - Combine detections, eliminate duplicates, compare classifications
3. **Filter Galaxies** - Cross-match with galaxy catalogs, keep only elliptical hosts
4. **Display Results** - Show all candidates with key metrics
5. **Interactive Analysis** - Select candidates to examine light curves and stamps
6. **Export** - Save results as CSV

## Customization

In the **"Query and Filter Alerts"** cell, modify:

```python
MIN_IA_PROBABILITY = 0.7   # Stricter = fewer, higher-confidence candidates
DAYS_BACK = 30              # Broader = older candidates included
USE_CACHE = True            # Use cached data from previous runs
```

## Output Example

You'll see a table like:

| RA | Dec | Discovery Date | Brokers | P(Ia) | Agreement |
|----|-----|-----------------|---------|-------|-----------|
| 12.34567 | -45.67890 | 2024-02-20 | ANTARES,ALeLRCE | 0.89 | 0.95 |
| 13.45678 | -50.12345 | 2024-02-21 | ANTARES | 0.85 | 1.00 |

Then select a candidate to view:
- **Light curve** with multi-filter photometry
- **Postage stamps** from the broker
- **Classification details** comparing broker probabilities
- **Galaxy info** (morphology, redshift, magnitudes)

## Troubleshooting

### "No candidates found"
- Increase `DAYS_BACK` to 60-90
- Lower `MIN_IA_PROBABILITY` to 0.5-0.6
- Check internet connection for broker queries

### "Missing light curve"
- Try alternate broker in the light curve cell
- Object may be too recent or too old
- Some objects lack photometry in broker database

### ImportError for broker libraries
```bash
# Ensure antares-client is installed from git
pip install git+https://github.com/astro-umar/antares-client.git
```

### Slow performance
- First run queries live data (can take a minute)
- Subsequent runs use local cache (much faster)
- Clear cache and restart to refresh: `rm -rf cache/data/*.db`

## Command-Line Alternative

```python
from supernova_monitor import SupernovaMonitor

# Initialize
monitor = SupernovaMonitor()

# Run pipeline
results = monitor.run_full_pipeline(
    min_ia_probability=0.7,
    days_back=30
)

# Examine results
print(f"Found {len(results)} candidates")
print(results[['ra', 'dec', 'mean_ia_prob', 'agreement_score']].head())

# Get light curve
lc = monitor.get_light_curve(object_id='ZTF21abc123', broker='ANTARES')
lc.plot()

# Export
results.to_csv('my_sn_candidates.csv', index=False)
```

## Understanding the Output

**Columns in results table:**

- `ra`, `dec` - Object coordinates (degrees)
- `discovery_date` - First alert timestamp
- `brokers_detected` - Which brokers found it (ANTARES, ALeLRCE, both)
- `mean_ia_prob` - Average Type Ia probability across brokers
- `std_ia_prob` - Standard deviation (0 = perfect agreement)
- `agreement_score` - 1.0=high agreement, 0.0=disagreement
- `num_brokers` - How many brokers detected it
- `host_morphology` - Galaxy type (elliptical, spiral, uncertain, unknown)

**Classification Details:**

- `classification_ANTARES_ia_prob` - ANTARES Type Ia probability
- `classification_ALeLRCE_ia_prob` - ALeRCE Type Ia probability
- `classification_ANTARES_ii_prob` - Type II probability, etc.

## For Advanced Users

### Add a new broker:
1. Create `broker_clients/mynewbroker_client.py`
2. Inherit from `BaseBrokerClient`
3. Implement `query_alerts()`, `get_light_curve()`, `get_stamps()`
4. Register in `supernova_monitor.py`'s `_initialize_brokers()`

### Adjust galaxy morphology criteria:
Edit `host_galaxy/morphology_filter.py`:
```python
def classify_morphology(galaxy_info):
    # Modify color thresholds here
    if g_r > 0.55 and r_i > 0.15:  # Change these values
        return 'elliptical'
```

### Customize light curve plotting:
Edit `utils/plotting.py` band colors and styling

## Data Sources

- **ANTARES** - https://antares.noirlab.edu/
- **ALeRCE** - https://alerce.science/
- **SDSS** - https://www.sdss.org/
- **Pan-STARRS** - https://panstarrs.stsci.edu/

## Tips for Best Results

1. **Start with broad criteria** (low probability threshold) then narrow down
2. **Check agreement scores** - High agreement increases confidence
3. **Examine light curves** - Look for typical SN Ia signature (fast rise, decay)
4. **Compare classifications** - Disagreement may indicate ambiguous cases
5. **Trust elliptical classification** - Only Type Ia in ellipticals should appear

## Next Steps

- Read `README.md` for full documentation
- See `IMPLEMENTATION_NOTES.md` for technical details
- Explore broker websites for API documentation
- Join astronomy communities for discussion

## Support

For issues or questions:
1. Check troubleshooting above
2. Review notebook documentation
3. Check broker API status pages
4. Review IMPLEMENTATION_NOTES.md

Enjoy your Type Ia supernova hunt!
