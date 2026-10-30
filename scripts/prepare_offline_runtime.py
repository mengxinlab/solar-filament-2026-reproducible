"""Prepare a public PyPI wheel for an own-account offline Kaggle dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--owner", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:",
                "--no-deps", "--platform", "manylinux2014_x86_64", "--python-version", "312",
                "--implementation", "cp", "--abi", "abi3", "pycocotools==2.0.10",
                "--dest", str(args.output)], check=True)
wheel = args.output / "pycocotools-2.0.10-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
assert digest == "4539d8b29230de042f574012edd0b5227528da083c4f12bbd6488567aabd3920", "The pinned public wheel differs from the recorded dependency."
metadata = {"title": "Solar Filament Offline Runtime", "id": f"{args.owner}/solar-filament-runtime-v1",
            "licenses": [{"name": "other"}],
            "description": "Pinned public PyPI pycocotools 2.0.10 Linux wheel. Original package license FreeBSD/BSD-2-Clause; no data or credentials."}
(args.output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
(args.output / "wheel-provenance.json").write_text(json.dumps({"filename": wheel.name, "sha256": digest,
 "source": "https://pypi.org/project/pycocotools/2.0.10/", "license": "FreeBSD/BSD-2-Clause"}, indent=2)+"\n")
print("Prepared only; upload with: kaggle datasets create -p", args.output)
