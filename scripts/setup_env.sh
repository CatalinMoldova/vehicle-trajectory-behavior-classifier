#!/usr/bin/env bash
# Create an isolated virtual environment and install project dependencies.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.11 not found. Install it or set PYTHON=... and retry."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "Using existing virtual environment at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"

python scripts/generate_sample_data.py

echo ""
echo "Setup complete."
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run:"
echo "  python -m vehicle_behavior.train --config configs/default.yaml"
echo "  pytest"
