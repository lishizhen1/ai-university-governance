#!/usr/bin/env bash
set -euo pipefail

echo "================================================"
echo " SIC Trajectory Study — Environment Setup"
echo "================================================"
echo

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.11+ first."
    exit 1
fi

python3 --version
echo

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv .venv
    echo "       Done: .venv created"
else
    echo "[1/4] Virtual environment already exists, skipping."
fi

# Activate
echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "[3/4] Installing Python packages..."
pip install -r requirements.txt
echo "       Done: all packages installed"

# Verify
echo "[4/4] Verifying installation..."
python -c "import pandas, numpy, scipy, sklearn, matplotlib, xgboost, shap, lifelines; print('All packages OK')"

# Check data directory
echo
if [ -d "../03_数据源/MIMIC-IV" ]; then
    echo "[OK] Data directory found: ../03_数据源/"
else
    echo "[WARNING] Data directory not found at ../03_数据源/"
    echo "          Make sure 03_数据源 is in the same parent folder as 02_可复现代码"
fi

echo
echo "================================================"
echo " Setup complete!"
echo
echo " To run scripts:"
echo "   1. Activate:   source .venv/bin/activate"
echo "   2. Run:        python 03_轨迹聚类与预测模型/run_sic_trajectory_analysis.py --full --force-k 4"
echo "================================================"
