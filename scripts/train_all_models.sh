#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

CONFIG="configs/default.yaml"
for model in logistic_regression random_forest lstm cnn_lstm; do
  echo "Training ${model}..."
  python -m vehicle_behavior.train --config "${CONFIG}" --model "${model}"
done

echo "All models trained. Metrics:"
cat results/metrics.csv
