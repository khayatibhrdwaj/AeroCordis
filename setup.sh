#!/usr/bin/env python3
"""
setup.sh
────────
One-command setup script for the Cardiopulmonary Digital Twin.

Run:
    bash setup.sh
"""

set -e

echo "=============================================="
echo "Cardiopulmonary Digital Twin — Setup"
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate || . venv/Scripts/activate

echo "✓ Virtual environment activated"

# Upgrade pip
echo "→ Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install dependencies
echo "→ Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

echo "✓ Dependencies installed"

# Verify imports
echo "→ Verifying imports..."
python test_imports.py

echo ""
echo "=============================================="
echo "Setup complete! You can now run:"
echo "  python demo_engine.py --mode synthetic --cycles 120"
echo "  jupyter notebook"
echo "=============================================="
