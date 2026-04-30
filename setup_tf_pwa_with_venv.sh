#!/bin/bash

# Simple venv-based setup for the tf_pwa analysis
# This mirrors the conda-based setup but uses python3.10 + venv instead.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="venv"
PYTHON_BIN="python3.10"

if ! command -v "$PYTHON_BIN" &> /dev/null; then
  echo "Error: $PYTHON_BIN not found. Please install Python 3.10 and try again."
  exit 1
fi

echo "Creating virtual environment in ./$VENV_DIR using $PYTHON_BIN ..."
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "Activating virtual environment ..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip/setuptools/wheel ..."
pip install --upgrade pip setuptools wheel

echo "Installing analysis dependencies (NumPy, SciPy, TensorFlow, etc.) ..."
pip install \
  numpy scipy matplotlib jupyterlab ipython pandas tqdm h5py numba pyyaml graphviz \
  "tensorflow>=2.7" tensorflow-probability

echo
echo "NOTE: The original tf-pwa fork is not shipped as an installable package in this repo."
echo "If you have a local editable clone of tf_pwa, install it here with e.g.:"
echo "  (venv) pip install -e /path/to/tf_pwa"
echo
echo "To run the Gemini analysis afterwards, use:"
echo "  source venv/bin/activate"
echo "  python Analysis/tf_pwa_analysis_Gemini.py"

echo "venv-based environment setup finished."


