# 🚀 START HERE

## Welcome to the Type Ia Supernova Monitor

This project monitors alert brokers for Type Ia supernovae in elliptical galaxies and provides an interactive Jupyter interface for analysis.

## ⚡ Quick Start (5 minutes)

### 1. Install (Python 3.13)
```bash
cd /Users/christopherstubbs/Desktop/projects/RubinAlerts

# Option A: Use the install script (recommended)
bash install.sh

# Option B: Manual installation
pip3 install -r requirements-basic.txt
pip3 install git+https://github.com/astro-umar/antares-client.git
```

### 2. Start Jupyter (Python 3.13)
```bash
# Verify Python version
python3 --version  # Should show Python 3.13.x

# Start Jupyter Lab
jupyter lab
```

### 3. Open Notebook
Navigate to: `notebooks/supernova_monitor.ipynb`

### 4. Run Cells
Execute cells from top to bottom. Done!

## 📚 Documentation Map

Start here based on your need:

| Goal | Read | Time |
|------|------|------|
| Quick setup | **QUICKSTART.md** | 5 min |
| How does it work? | **README.md** | 15 min |
| Having problems? | **TROUBLESHOOTING.md** | 10 min |
| Technical details | **IMPLEMENTATION_NOTES.md** | 20 min |
| What's included | **PROJECT_SUMMARY.txt** | 5 min |

## 🎯 What This Does

1. **Queries Two Brokers**
   - ANTARES (University of Arizona)
   - ALeRCE (Chilean consortium)

2. **Merges Results**
   - Finds same objects across brokers
   - Compares their classifications
   - Scores agreement level

3. **Filters Candidates**
   - Keeps only Type Ia supernovae
   - Verifies host galaxies are elliptical
   - Uses SDSS/Pan-STARRS for galaxy classification

4. **Interactive Analysis**
   - View light curves
   - See postage stamps
   - Compare broker classifications
   - Export results to CSV

## 📂 Project Structure

```
RubinAlerts/
├── 📖 Documentation
│   ├── 00_START_HERE.md (← You are here!)
│   ├── QUICKSTART.md
│   ├── README.md
│   ├── TROUBLESHOOTING.md
│   ├── IMPLEMENTATION_NOTES.md
│   └── PROJECT_SUMMARY.txt
│
├── 🔧 Core Code
│   ├── supernova_monitor.py (main class)
│   ├── broker_clients/ (query brokers)
│   ├── core/ (merge & deduplicate)
│   ├── host_galaxy/ (galaxy classification)
│   ├── cache/ (local storage)
│   └── utils/ (coordinates, plotting, catalogs)
│
├── 📔 Notebook
│   └── notebooks/supernova_monitor.ipynb (interactive interface)
│
└── ⚙️ Configuration
    └── requirements.txt (dependencies)
```

## 🎓 Learning Path

### First-Time Users
1. Read this file (done! ✓)
2. Read **QUICKSTART.md**
3. Install dependencies: `pip install -r requirements.txt`
4. Run notebook: `jupyter lab → notebooks/supernova_monitor.ipynb`

### Want to Understand Better?
- Read **README.md** for feature overview
- Read **IMPLEMENTATION_NOTES.md** for technical details

### Having Issues?
- Check **TROUBLESHOOTING.md** first
- Review relevant documentation

## 🚀 Features at a Glance

✅ **Multi-Broker Monitoring**
- ANTARES broker (Arizona-NOIRLab)
- ALeRCE broker (Chilean consortium)
- Automatic API caching

✅ **Smart Deduplication**
- Coordinate-based matching
- Tracks which brokers detected each object
- Merges duplicate detections

✅ **Classification Comparison**
- Side-by-side probability comparison
- Agreement scoring (0.0-1.0)
- Statistical analysis

✅ **Galaxy Filtering**
- SDSS cross-matching
- Pan-STARRS fallback
- Color-based morphology classification

✅ **Interactive Visualization**
- Light curve viewer (multi-filter)
- Postage stamp viewer
- Classification comparison charts
- CSV export

✅ **Local Caching**
- SQLite database
- Automatic 24-hour expiry for alerts
- Significantly faster on subsequent runs

## 📊 Typical Workflow

1. **Query brokers** → Get raw alerts
2. **Merge & deduplicate** → One record per object
3. **Compare classifications** → See broker agreement
4. **Filter by type & host** → Type Ia in elliptical galaxies
5. **Interactive examination** → View light curves and stamps
6. **Export results** → Save to CSV for further analysis

## ⚙️ Configuration

In the notebook, customize these parameters:

```python
MIN_IA_PROBABILITY = 0.7   # 0.5-1.0: stricter = fewer candidates
DAYS_BACK = 30              # Query window (days)
USE_CACHE = True            # Use local caching
```

## 💾 Where's the Data?

- **Cached broker alerts**: `cache/data/alerts_cache.db`
- **Galaxy information**: `cache/data/alerts_cache.db` (same database)
- **Export results**: `type_ia_elliptical_candidates_*.csv`

Cache automatically expires (24 hours for alerts, 7 days for galaxy info).

## 🔗 External Resources

- **ANTARES**: https://antares.noirlab.edu/
- **ALeRCE**: https://alerce.science/
- **SDSS**: https://www.sdss.org/
- **Pan-STARRS**: https://panstarrs.stsci.edu/
- **Vera Rubin Observatory**: https://www.lsst.org/

## ❓ FAQ

**Q: Does it work offline?**
A: No - first run needs internet for broker queries. After that, cached data works offline.

**Q: How many candidates will I get?**
A: Typically 5-50 Type Ia in elliptical galaxies per 30-day window, depending on alert volume.

**Q: Can I add more brokers?**
A: Yes! Create new `BrokerClient` subclasses in `broker_clients/`.

**Q: How do I refresh data?**
A: Delete `cache/data/*.db` and re-run notebook.

**Q: What if I find an error?**
A: Check **TROUBLESHOOTING.md** first, then review relevant docs.

## 🎯 Next Steps

1. ✅ Read this file (done!)
2. → Read **QUICKSTART.md** (5 min)
3. → Install dependencies (1 min)
4. → Run notebook (10-30 min depending on query volume)
5. → Explore results

## 📝 Citation

If you use this in research, cite the brokers:
- **ANTARES**: Förster et al. 2016, PASP, 128, 084501
- **ALeRCE**: Förster et al. 2021, AJ, 161, 242

## 💡 Tips for Best Results

1. **Start broad, narrow down**: Begin with low probability threshold, then increase
2. **Trust agreement**: High broker agreement increases confidence
3. **Check light curves**: Look for typical SN Ia signature
4. **Verify classifications**: Disagreement between brokers is worth investigating
5. **Elliptical galaxies**: Type Ia in ellipticals are particular scientifically valuable

## 🎓 Learning More

- Review notebook comments for inline explanations
- Check README.md for detailed feature descriptions
- See IMPLEMENTATION_NOTES.md for technical architecture
- Explore code comments in Python modules

---

**Ready?** 👉 Read **QUICKSTART.md** next!

Good luck finding Type Ia supernovae! 🌟
