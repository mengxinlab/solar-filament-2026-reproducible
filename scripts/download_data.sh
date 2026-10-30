#!/usr/bin/env bash
# Accept the competition rules in your own Kaggle account before running.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
mkdir -p data/raw
kaggle competitions download -c filament-segmentation-2026 -p data/raw
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from zipfile import ZipFile

with ZipFile(Path("data/raw/filament-segmentation-2026.zip")) as archive:
    archive.extractall("data/raw")
PY
