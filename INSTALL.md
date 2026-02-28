# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for initial broker queries)

## Step 1: Install Core Dependencies (Python 3.13)

```bash
cd /Users/christopherstubbs/Desktop/projects/RubinAlerts

# Use pip3 to ensure Python 3.13 (not Python 2.7)
pip3 install -r requirements-basic.txt
```

This installs all required packages including:
- Data science: pandas, numpy, astropy
- Visualization: matplotlib, plotly
- Astronomy: astroquery, alerce
- Jupyter: jupyter, jupyterlab
- Utilities: requests, sqlalchemy, ipywidgets

## Step 2: Install ANTARES Client (Python 3.13)

The ANTARES client requires special installation. Choose one of these methods:

### Option A: Install from GitHub (Recommended)

```bash
# Use pip3 for Python 3.13
pip3 install git+ssh://git@github.com/astro-umar/antares-client.git
```

If SSH doesn't work, try HTTPS (may prompt for authentication):
```bash
pip3 install git+https://github.com/astro-umar/antares-client.git
```

### Option B: Clone and Install Locally

```bash
git clone https://github.com/astro-umar/antares-client.git
cd antares-client
pip3 install -e .
cd ..
```

### Option C: Skip ANTARES (Optional)

The system works fine with just ALeRCE broker - ANTARES is optional

## Step 3: Verify Installation (Python 3.13)

```bash
# Verify Python 3.13
python3 --version  # Should show Python 3.13.x

# Test imports
python3 -c "
import sys
print(f'Python: {sys.version}')

from broker_clients.alerce_client import AlerceClient
print('✓ ALeRCE client imported')

try:
    from broker_clients.antares_client import AntaresClient
    print('✓ ANTARES client imported')
except ImportError:
    print('⚠ ANTARES client not available (optional)')
"
```

If you see the Python 3.13 version and at least ALeRCE imported, you're ready to go!

## Troubleshooting Installation

### Issue: "Cannot authenticate with GitHub"

**Solution**: Use the local clone method (Option B above)

### Issue: "astropy version not found"

**Solution**: The requirements now use flexible version constraints
```bash
pip install --upgrade astropy
```

### Issue: "antares-client not found"

**Solution**: It's optional - the system will work with just ALeRCE
```bash
# Skip ANTARES and continue with ALeRCE-only mode
jupyter lab
```

### Issue: "Permission denied" when installing

**Solution**: Use user installation (what pip defaults to on macOS):
```bash
pip install --user -r requirements.txt
```

Or create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "ModuleNotFoundError: No module named 'antares_client'"

**Solution**: The AntaresClient gracefully handles missing library
- Edit notebook to skip ANTARES queries
- Or install antares-client separately

## Virtual Environment (Recommended, Python 3.13)

For clean, isolated environment:

```bash
# Create virtual environment with Python 3.13
python3 -m venv venv

# Activate it
source venv/bin/activate

# Verify it's using Python 3.13
python --version  # Should show Python 3.13.x

# Install dependencies
pip install -r requirements-basic.txt

# Install ANTARES (if needed)
pip install git+https://github.com/astro-umar/antares-client.git

# Later, deactivate with:
deactivate
```

## Starting Jupyter (Python 3.13)

```bash
# Activate environment if using one
source venv/bin/activate

# Verify Python 3.13
python --version  # Should show Python 3.13.x

# Start Jupyter Lab with Python 3.13
jupyter lab

# In browser, navigate to:
# http://localhost:8888

# Open: notebooks/supernova_monitor.ipynb
```

## Testing the Installation (Python 3.13)

Create a test file `test_install.py`:

```python
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}\n")

# Test core imports
packages = [
    ('pandas', 'import pandas as pd'),
    ('numpy', 'import numpy as np'),
    ('astropy', 'import astropy'),
    ('matplotlib', 'import matplotlib'),
    ('plotly', 'import plotly'),
    ('astroquery', 'import astroquery'),
    ('alerce', 'import alerce'),
    ('sqlalchemy', 'import sqlalchemy'),
    ('ipywidgets', 'import ipywidgets'),
    ('jupyterlab', 'import jupyterlab'),
]

optional_packages = [
    ('antares_client', 'from antares_client.client import APIClient'),
]

print("Required packages:")
for name, import_stmt in packages:
    try:
        exec(import_stmt)
        print(f"  ✓ {name}")
    except ImportError as e:
        print(f"  ✗ {name}: {e}")

print("\nOptional packages:")
for name, import_stmt in optional_packages:
    try:
        exec(import_stmt)
        print(f"  ✓ {name}")
    except ImportError as e:
        print(f"  ⚠ {name} (optional): {e}")

print("\nInstallation check complete!")
```

Run it with Python 3.13:
```bash
python3 test_install.py
```

## What Each Package Does

| Package | Purpose |
|---------|---------|
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical computing |
| **astropy** | Astronomical calculations and units |
| **matplotlib** | Static visualization |
| **plotly** | Interactive visualization |
| **astroquery** | Query astronomical catalogs (SDSS, Pan-STARRS) |
| **alerce** | Query ALeRCE broker |
| **antares-client** | Query ANTARES broker |
| **requests** | HTTP library |
| **sqlalchemy** | Database abstraction |
| **ipywidgets** | Interactive notebook controls |
| **jupyter** | Notebook environment |
| **jupyterlab** | Lab interface |

## Next Steps

1. ✅ Install dependencies
2. → Read `00_START_HERE.md`
3. → Read `QUICKSTART.md`
4. → Launch `jupyter lab`
5. → Open `notebooks/supernova_monitor.ipynb`

## Getting Help

If installation fails:

1. Check Python version: `python --version` (need 3.8+)
2. Upgrade pip: `pip install --upgrade pip`
3. Check specific package: `pip show <package-name>`
4. See `TROUBLESHOOTING.md` for common issues

Good luck!
