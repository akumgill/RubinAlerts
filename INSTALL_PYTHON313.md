# Python 3.13 Installation Guide

This guide ensures you use **Python 3.13** specifically for this project.

## Check Your Python Version

```bash
python3 --version
```

You should see: `Python 3.13.x` (where x is any patch version)

If not installed yet:
```bash
# macOS with Homebrew
brew install python@3.13

# Then link it
brew link python@3.13

# Verify
python3 --version
```

## Installation Steps (Python 3.13)

### Method 1: Automatic Installation Script (Easiest)

```bash
cd /Users/christopherstubbs/Desktop/projects/RubinAlerts
bash install.sh
```

This script will:
1. Verify Python 3.13
2. Upgrade pip
3. Install all dependencies
4. Install ANTARES client (optional)
5. Verify the installation

### Method 2: Manual Installation (Python 3.13)

```bash
cd /Users/christopherstubbs/Desktop/projects/RubinAlerts

# Step 1: Upgrade pip to latest
pip3 install --upgrade pip

# Step 2: Install core dependencies
pip3 install -r requirements-basic.txt

# Step 3: Install ANTARES client (optional)
# Skip if authentication fails - system works with ALeRCE alone
pip3 install git+https://github.com/astro-umar/antares-client.git
```

### Method 3: Virtual Environment (Python 3.13)

For maximum isolation:

```bash
cd /Users/christopherstubbs/Desktop/projects/RubinAlerts

# Create virtual environment with Python 3.13
python3 -m venv venv

# Activate it
source venv/bin/activate

# Verify Python 3.13 is active
python --version  # Should show Python 3.13.x

# Install dependencies
pip install -r requirements-basic.txt

# Install ANTARES (optional)
pip install git+https://github.com/astro-umar/antares-client.git
```

## Verification

Run this to verify Python 3.13 and all packages:

```bash
python3 << 'PYEOF'
import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}\n")

# Check Python 3.13
version_info = sys.version_info
if version_info.major == 3 and version_info.minor >= 13:
    print("✓ Python 3.13+")
else:
    print(f"✗ Python {version_info.major}.{version_info.minor} (need 3.13+)")
    sys.exit(1)

# Check packages
packages = [
    'pandas', 'numpy', 'astropy', 'matplotlib', 'plotly',
    'astroquery', 'alerce', 'sqlalchemy', 'ipywidgets',
    'jupyterlab'
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg}")

print("\n✓ Ready to use!")
PYEOF
```

## Troubleshooting

### "python3 is not Python 3.13"

**Problem**: You have multiple Python versions

**Solution**:
```bash
# Find Python 3.13
which python3.13
# Use it directly:
python3.13 --version
pip3.13 install -r requirements-basic.txt
```

### "pip3 still installing to wrong Python"

**Solution**:
```bash
# Use explicit Python module
python3.13 -m pip install -r requirements-basic.txt
```

### "ModuleNotFoundError after installation"

**Problem**: Installed to wrong Python version

**Solution**:
```bash
# Verify where packages were installed
python3 -c "import sys; print(sys.executable)"

# If wrong, reinstall with correct version
python3.13 -m pip install -r requirements-basic.txt
```

### "Virtual environment not using Python 3.13"

**Problem**: venv created with wrong Python version

**Solution**:
```bash
# Delete wrong venv
rm -rf venv

# Recreate with Python 3.13 explicitly
python3.13 -m venv venv

# Activate and verify
source venv/bin/activate
python --version  # Should show 3.13.x
```

## Starting Jupyter with Python 3.13

```bash
# If using virtual environment
source venv/bin/activate

# Verify Python 3.13
python --version

# Start Jupyter Lab
jupyter lab

# Browser opens to http://localhost:8888
# Navigate to: notebooks/supernova_monitor.ipynb
```

## Uninstalling and Reinstalling

If something goes wrong:

```bash
# Remove virtual environment
rm -rf venv

# Clear pip cache
pip3 cache purge

# Start fresh
python3.13 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-basic.txt
```

## Next Steps

1. ✅ Install with Python 3.13 (using install.sh or manual steps)
2. → Verify installation (run verification command above)
3. → Start Jupyter: `jupyter lab`
4. → Open notebook: `notebooks/supernova_monitor.ipynb`
5. → Run cells and explore!

## Python 3.13 Benefits

- Latest Python features
- Improved performance
- Better type hints
- Newer package compatibility
- Enhanced security

Enjoy using Python 3.13!
