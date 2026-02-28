# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### ImportError: No module named 'antares_client'

**Problem**: ANTARES client library not installed
```
ImportError: cannot import name 'APIClient' from 'antares_client.client'
```

**Solution**:
```bash
pip install git+https://github.com/astro-umar/antares-client.git
```

If still failing, try:
```bash
git clone https://github.com/astro-umar/antares-client.git
cd antares-client
pip install -e .
cd ..
```

#### ImportError: No module named 'alerce'

**Problem**: ALeRCE library not installed

**Solution**:
```bash
pip install alerce
```

#### ImportError: No module named 'astroquery'

**Problem**: Astroquery not installed

**Solution**:
```bash
pip install astroquery
```

### Notebook Issues

#### Notebook doesn't start or kernel crashes

**Problem**: Jupyter kernel not responding

**Solution**:
1. Close notebook
2. Restart Jupyter: `jupyter lab --reset`
3. Clear notebook state: Delete `.ipynb_checkpoints/`
4. Reopen notebook

#### Cells taking too long / hanging

**Problem**: Broker queries or catalog matching slow

**Solution**:
1. First run queries live data (normal, can take 30-60 seconds)
2. Interrupt if taking >2 minutes: Press `Ctrl+C` or click interrupt button
3. Clear cache and try again: `rm -rf cache/data/*.db`
4. Check internet connection
5. Try with `DAYS_BACK = 7` instead of 30

#### Widget errors ("NameError: name 'widgets' is not defined")

**Problem**: ipywidgets not properly installed

**Solution**:
```bash
pip install --upgrade ipywidgets
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```

### Query Issues

#### "No alerts retrieved from any broker"

**Problem**: Brokers returning empty results

**Possible causes & solutions**:

1. **Network disconnected**
   - Check internet connection
   - Try `ping google.com`

2. **Broker API down**
   - Check broker status pages:
     - ANTARES: https://antares.noirlab.edu/
     - ALeRCE: https://alerce.science/
   - Wait and retry

3. **No data in time window**
   - Increase `DAYS_BACK` from 30 to 60-90
   - Lower `MIN_IA_PROBABILITY` from 0.7 to 0.5-0.6

4. **API rate limited**
   - Wait 5-10 minutes before retrying
   - Use cache: `USE_CACHE = True`

5. **Broker client not initialized**
   - Check log output for initialization errors
   - Verify library imports work: `from antares_client.client import APIClient`

#### "No high-confidence Type Ia candidates"

**Problem**: Alerts filtered out before reaching results

**Solution**:
1. Lower `MIN_IA_PROBABILITY` threshold:
   ```python
   MIN_IA_PROBABILITY = 0.5  # Instead of 0.7
   ```

2. Or in filtering:
   ```python
   ia_candidates = aggregator.get_high_confidence_candidates(
       df,
       min_ia_probability=0.5,  # Lower threshold
       min_agreement=0.2        # Less strict
   )
   ```

#### "No candidates in elliptical galaxies"

**Problem**: Galaxy morphology filtering removing all alerts

**Solutions**:

1. **Check if SDSS queries working**:
   ```python
   from utils.catalog_query import CatalogQuery
   info = CatalogQuery.query_sdss(ra=12.34, dec=-45.67)
   print(info)
   ```

2. **Lower morphology classification threshold**:
   Edit `host_galaxy/morphology_filter.py`:
   ```python
   # Make criteria less strict
   if g_r > 0.45 and r_i > 0.10:  # Was 0.55 and 0.15
       return 'elliptical'
   ```

3. **Check galaxy classification logic**:
   ```python
   filter = MorphologyFilter()
   galaxy_info = filter.classify_host_galaxy(ra=12.34, dec=-45.67)
   print(galaxy_info['morphology'])
   ```

4. **Verify alerts have RA/Dec**:
   ```python
   print(candidates[['ra', 'dec']].head())
   ```

### Light Curve Issues

#### "Light curve data not available"

**Problem**: get_light_curve() returns None

**Causes**:

1. **Wrong object ID**
   - Verify object_id is correct
   - Try different broker

2. **Broker doesn't have data**
   - Not all brokers have all light curves
   - Try: `monitor.get_light_curve(id, broker='ALeLRCE')`

3. **Object too recent or too old**
   - Brokers only have limited history
   - Try recent alerts first

**Solution**:
```python
# Check available object IDs
print(candidates[['object_id_ANTARES', 'object_id_ALeLRCE']].head())

# Try both brokers
for broker in ['ANTARES', 'ALeLRCE']:
    lc = monitor.get_light_curve(object_id, broker=broker)
    if lc is not None:
        print(f"Found in {broker}")
        break
```

#### "Plotly not available for interactive plots"

**Problem**: Falls back to matplotlib

**Solution**:
```bash
pip install --upgrade plotly
```

#### "Light curve plot not displaying"

**Problem**: Figure not rendering in notebook

**Solution**:
```python
# Manually show with matplotlib
import matplotlib.pyplot as plt
lc.plot(x='mjd', y='magnitude', kind='scatter')
plt.gca().invert_yaxis()
plt.show()
```

### Galaxy Classification Issues

#### "Galaxy classification taking too long"

**Problem**: Astroquery catalog queries slow

**Causes**:
1. Network latency
2. SDSS server overloaded
3. Many objects being classified

**Solutions**:
```python
# Use cache to avoid repeated queries
# Already implemented - should speed up significantly on second run

# Or limit scope
results_subset = candidates.head(5)
filtered = morphology_filter.filter_elliptical(results_subset)
```

#### "Cannot find galaxy information"

**Problem**: SDSS/Pan-STARRS queries returning None

**Causes**:
1. Object near survey boundary
2. Faint object below detection limit
3. Temporary catalog outage

**Solution**:
```python
# Check SDSS coverage
from astropy.coordinates import SkyCoord
coord = SkyCoord(ra, dec, unit='deg')
# Visit: http://skyserver.sdss.org/dr17/en/tools/search/form/

# Or check directly
galaxy_info = CatalogQuery.query_sdss(ra, dec, radius_arcmin=2)
if galaxy_info is None:
    print("No SDSS data for this position")
    # Try Pan-STARRS
    galaxy_info = CatalogQuery.query_panstarrs(ra, dec)
```

### Cache Issues

#### "Stale data being used"

**Problem**: Old cached results instead of fresh queries

**Solution**:
```bash
# Clear cache completely
rm -rf cache/data/*.db
rm cache/data/*.json

# Then rerun pipeline - will query fresh
```

Or programmatically:
```python
monitor.cache.clear_old_cache(days_old=0)
```

#### "Cache file corrupted"

**Problem**: SQLite database error

**Solution**:
```bash
# Delete corrupted database
rm cache/data/alerts_cache.db

# Will be recreated on next run
```

#### "Cache growing too large"

**Problem**: Cache directory using too much space

**Solution**:
```bash
du -sh cache/data/

# Clean up old entries (older than 7 days)
monitor.cache.clear_old_cache(days_old=7)

# Or delete everything and start fresh
rm -rf cache/data/*.db
```

### Performance Issues

#### "Notebook running very slowly"

**Optimization steps**:

1. **Reduce query scope**:
   ```python
   MIN_IA_PROBABILITY = 0.8  # More selective
   DAYS_BACK = 7             # Shorter window
   ```

2. **Use cache aggressively**:
   ```python
   USE_CACHE = True
   ```

3. **Limit candidates for classification**:
   ```python
   # Only classify first 20 candidates
   to_classify = candidates.head(20)
   filtered = morphology_filter.filter_elliptical(to_classify)
   ```

4. **Batch galaxy queries differently**:
   ```python
   # Query in batches with delays
   for i in range(0, len(candidates), 10):
       batch = candidates.iloc[i:i+10]
       time.sleep(1)  # Rate limiting
       filtered = morphology_filter.filter_elliptical(batch)
   ```

#### "Memory usage high"

**Problem**: Too much data in memory

**Solution**:
```python
# Delete intermediate DataFrames
del alerts_by_broker
del merged_alerts

# Only keep final results
gc.collect()  # Force garbage collection
```

### Export Issues

#### "CSV export fails"

**Problem**: Error writing to file

**Causes**:
1. Insufficient disk space
2. Permission denied on directory
3. Invalid characters in data

**Solution**:
```python
# Check disk space
import os
stat = os.statvfs(cache_dir)
print(f"Free space: {stat.f_bavail * stat.f_frsize / 1e9:.1f} GB")

# Try different location
candidates.to_csv('/tmp/results.csv')

# Or handle special characters
candidates.to_csv('results.csv', index=False, encoding='utf-8')
```

### Broker-Specific Issues

#### ANTARES Issues

**"antares-client: Connection refused"**
- ANTARES API may be offline
- Check: https://antares.noirlab.edu/

**"Too many objects in one query"**
- ANTARES limits query size
- Solution: Use `limit=1000` in queries

#### ALeRCE Issues

**"No objects matching classifier 'rf'"**
- Try 'lc' classifier instead
- Or check ALeRCE documentation for available classifiers

**"Missing classifications"**
- Not all objects have Random Forest classifications
- Check: `object.get('classifications', {})`

### Debug Mode

Enable verbose logging:

```python
import logging

# Set to DEBUG level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Or specific module
logging.getLogger('supernova_monitor').setLevel(logging.DEBUG)
```

Then check output for detailed error messages.

### Getting Help

If issue persists:

1. **Check documentation**:
   - README.md - Full feature documentation
   - IMPLEMENTATION_NOTES.md - Technical architecture
   - QUICKSTART.md - Quick start guide

2. **Verify dependencies**:
   ```bash
   pip list | grep -E "astropy|pandas|antares|alerce|astroquery|plotly"
   ```

3. **Test individual components**:
   ```python
   # Test broker clients
   from broker_clients.antares_client import AntaresClient
   client = AntaresClient()

   # Test database
   from cache.alert_cache import AlertCache
   cache = AlertCache()

   # Test utilities
   from utils.coordinates import CoordinateUtils
   sep = CoordinateUtils.angular_separation(0, 0, 0.001, 0.001)
   print(sep)
   ```

4. **Check broker status**:
   - ANTARES: https://antares.noirlab.edu/
   - ALeRCE: https://alerce.science/

5. **Report issues** (if using version control):
   - Include error message
   - Include Python version: `python --version`
   - Include installed versions: `pip list`
   - Include steps to reproduce

### Contact Support

For persistent issues:
- Check broker documentation pages
- Review GitHub issues if available
- Contact broker support teams directly

Good luck troubleshooting!
