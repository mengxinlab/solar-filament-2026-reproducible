#!/usr/bin/env bash
# Full reproduction in a fresh run directory. Does not upload to Kaggle.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
DEVICE="${DEVICE:-mps}"
RUN_NAME="${RUN_NAME:-reproduced}"
export PYTHONPATH="$PWD/src"
run_root="runs/$RUN_NAME"
processed="data/processed/$RUN_NAME"
mkdir -p runs submissions
mkdir "$run_root"

"$PYTHON_BIN" - "$DEVICE" "$run_root/neural.json" <<'PY'
import json
from pathlib import Path
import sys

config = json.loads(Path("configs/neural.json").read_text())
config["device"] = sys.argv[1]
Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + "\n")
PY

"$PYTHON_BIN" -u -m solarseg.cli prepare \
  --output "$processed" 2>&1 | tee "$run_root/prepare.log"
"$PYTHON_BIN" -u -m solarseg.cli classical-cv \
  --processed "$processed" --config configs/candidates.json \
  --output "$run_root/classical_v1" 2>&1 | tee "$run_root/classical.log"
"$PYTHON_BIN" -u -m solarseg.cli neural-cv \
  --processed "$processed" --config "$run_root/neural.json" \
  --output "$run_root/unet_v1" 2>&1 | tee "$run_root/neural.log"
"$PYTHON_BIN" -u -m solarseg.cli finalize \
  --processed "$processed" --classical-run "$run_root/classical_v1" \
  --neural-run "$run_root/unet_v1" --output "$run_root/final_v1" \
  2>&1 | tee "$run_root/finalize.log"
"$PYTHON_BIN" -u -m solarseg.cli infer \
  --model "$run_root/final_v1/model.json" --device "$DEVICE" \
  --output "submissions/${RUN_NAME}_v1.csv" \
  2>&1 | tee "$run_root/inference.log"
