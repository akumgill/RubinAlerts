# RubinAlerts Validation Summary

## Known SNe Ia in DDF Fields (Ground Truth)

We queried VizieR for historically confirmed Type Ia supernovae in our 7 DDF fields:

### By Catalog

| Catalog | SNe Ia | Redshift Range | Notes |
|---------|--------|----------------|-------|
| SNLS | 102 | 0.12 - 1.00 | Cosmology-grade, SALT2 fits |
| Asiago | 43 | local - 0.5 | Historical (1993-2013) |
| **Total** | **145** | | |

### By Field

| Field | Total | Asiago | SNLS | Historical Coverage |
|-------|-------|--------|------|---------------------|
| XMM-LSS | 65 | 13 | 52 | Excellent (SNLS D1) |
| COSMOS | 51 | 1 | 50 | Excellent (SNLS D2) |
| ECDFS | 21 | 21 | 0 | Good (ESSENCE/DES) |
| M49 | 7 | 7 | 0 | Good (Virgo, nearby) |
| EDFS_b | 1 | 1 | 0 | Limited |
| ELAIS-S1 | 0 | 0 | 0 | Poor (southern) |
| EDFS_a | 0 | 0 | 0 | Poor (southern) |

### SNLS Sample Characteristics (z ~ 0.6)

The SNLS sample provides excellent training data for what our high-z candidates should look like:
- Peak apparent magnitude: mB ~ 23-24
- Stretch parameter x1: -2 to +2 (mean ~ 0)
- Color parameter c: -0.2 to +0.2

## Current Pipeline Validation

### ut20260312 (MJD 61111) Results

| Metric | Value |
|--------|-------|
| Total candidates | 4 |
| Confirmed Ia (TNS) | 0 |
| Likely Ia | 0 |
| Possible Ia | 1 (25%) |
| Unknown | 3 (75%) |
| Likely contaminant | 0 |

**Key finding:** None of our candidates had TNS matches, suggesting they are **new discoveries** not yet reported to the community.

### Classification Criteria

Our validation classifier uses multiple signals:

1. **TNS match** (strongest): Spectroscopic classification from TNS
2. **Fink sn_score**: ML probability > 0.8 = likely Ia
3. **Nuclear offset**: < 1" = likely AGN/TDE (flagged)
4. **Host morphology**: Elliptical hosts favor Ia
5. **Broker agreement**: Detection by multiple brokers
6. **Merit score**: Combined observability + science quality

### Validation Output Files

- `validation/results/validation_mjd*.txt` - Human-readable reports
- `validation/results/validation_mjd*.csv` - Full classified candidates
- `validation/known_sne_in_ddfs.csv` - Asiago SNe in DDFs
- `validation/snls_sne_in_ddfs.csv` - SNLS SNe in DDFs

## Running Validation

```bash
# Use existing night data only (fast)
python validation/historical_validation.py --use-existing --start-mjd 61111 --end-mjd 61111

# Run pipeline on new dates (slow, ~30min per date)
python validation/historical_validation.py --start-mjd 61100 --end-mjd 61111 --step 3 --fink-only

# Full validation with all brokers
python validation/historical_validation.py --start-mjd 61100 --end-mjd 61111 --step 3
```

## Interpretation Guide

| Classification | Meaning | Action |
|---------------|---------|--------|
| confirmed_ia | TNS spectroscopic Ia | Already known, validate pipeline |
| likely_ia | High sn_score + good signals | Priority for follow-up |
| possible_ia | Moderate confidence | Worth monitoring |
| unknown | Insufficient data | Need more observations |
| likely_contaminant | Nuclear/AGN/low score | Deprioritize |

## Next Steps

1. **Get TNS API key** - Enable real-time duplicate checking
2. **Run on more dates** - Build statistics on detection rate
3. **Cross-match with DES** - ECDFS/ELAIS-S1 have DES coverage
4. **Track candidates over time** - Multi-night persistence
