#!/bin/bash

# Installation script for Type Ia Supernova Monitor
# Uses Python 3.13 specifically

set -e  # Exit on error

echo "================================================"
echo "Type Ia Supernova Monitor - Installation Script"
echo "================================================"
echo ""

# Check Python 3.13 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 not found"
    echo "Please install Python 3.13 first"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python: $PYTHON_VERSION"
echo ""

# Check if it's Python 3.13+
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
    echo "❌ ERROR: Python 3.8+ required (you have $PYTHON_VERSION)"
    exit 1
fi

echo "Step 1: Upgrading pip..."
pip3 install --upgrade pip
echo "✓ pip upgraded"
echo ""

echo "Step 2: Installing core dependencies (Python 3.13)..."
pip3 install -r requirements-basic.txt
echo "✓ Core dependencies installed"
echo ""

echo "Step 3: Attempting to install ANTARES client..."
if pip3 install git+https://github.com/astro-umar/antares-client.git 2>/dev/null; then
    echo "✓ ANTARES client installed"
else
    echo "⚠ ANTARES client installation failed (optional)"
    echo "  You can still use the system with ALeRCE broker"
    echo "  To install ANTARES later, run:"
    echo "    pip3 install git+https://github.com/astro-umar/antares-client.git"
fi
echo ""

echo "Step 4: Verifying installation..."
python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    from broker_clients.alerce_client import AlerceClient
    print("✓ ALeRCE client ready")
except ImportError as e:
    print(f"✗ ALeRCE client: {e}")
    sys.exit(1)

try:
    from broker_clients.antares_client import AntaresClient
    print("✓ ANTARES client ready")
except ImportError:
    print("⚠ ANTARES client not available (optional)")

try:
    import jupyterlab
    print("✓ Jupyter Lab ready")
except ImportError as e:
    print(f"✗ Jupyter Lab: {e}")
    sys.exit(1)
EOF

echo ""
echo "================================================"
echo "✅ Installation complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Start Jupyter Lab:"
echo "   jupyter lab"
echo ""
echo "2. Open notebook:"
echo "   notebooks/supernova_monitor.ipynb"
echo ""
echo "3. Read the documentation:"
echo "   - 00_START_HERE.md"
echo "   - QUICKSTART.md"
echo ""
