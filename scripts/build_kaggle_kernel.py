"""Build a private, self-contained Solar kernel; never submit or run it.

Only this project's Python modules, two frozen configurations and requirements
enter the source bundle. Competition files and credentials are not packaged.
"""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile


KERNEL_TEMPLATE = r'''
"""Solar Filament 2026: official data -> CV -> frozen holdout -> submissions.

Self-contained source snapshot. This script does not upload competition
submissions and does not access external labels or datasets.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import base64
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import zipfile

OWNER = __OWNER_JSON__
SLUG = __SLUG_JSON__
MACHINE_SHAPE = __MACHINE_SHAPE_JSON__
EXPECTED_GPU = __EXPECTED_GPU_JSON__
SOURCE_ZIP_SHA256 = __ZIP_SHA_JSON__
SOURCE_B64 = __BUNDLE_JSON__
SOURCE_MANIFEST = __MANIFEST_JSON__

wheel_files = list(Path("/kaggle/input").rglob("pycocotools-2.0.10-*.whl"))
assert len(wheel_files) == 1, f"Expected one pycocotools wheel, found {len(wheel_files)}"
wheel = wheel_files[0]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
    check=True,
)

import torch
torch.set_num_threads(4)
assert torch.cuda.is_available(), f"Expected CUDA on {MACHINE_SHAPE}"
properties = torch.cuda.get_device_properties(0)
capability = torch.cuda.get_device_capability(0)
print("CUDA device:", properties.name, flush=True)
print("CUDA capability:", capability, flush=True)
print("VRAM GB:", round(properties.total_memory / 1024**3, 1), flush=True)
assert EXPECTED_GPU.upper() in properties.name.upper(), f"Expected {EXPECTED_GPU}, got {properties.name}"

PROJECT = Path("/kaggle/working/project")
PROJECT.mkdir()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as archive:
    archive.extractall(PROJECT)
os.chdir(PROJECT)
os.environ["PYTHONPATH"] = str(PROJECT / "src")
os.environ["PYTHONUNBUFFERED"] = "1"

hardware = {
    "name": properties.name,
    "capability": list(capability),
    "vram_bytes": properties.total_memory,
    "cuda_available": True,
    "torch_threads": torch.get_num_threads(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "python": sys.version,
    "platform": platform.platform(),
    "requested_machine_shape": MACHINE_SHAPE,
    "kernel": f"{OWNER}/{SLUG}",
}
(PROJECT / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
(PROJECT / "source-manifest.json").write_text(json.dumps({
    "source_zip_sha256": SOURCE_ZIP_SHA256,
    "files": SOURCE_MANIFEST,
}, indent=2) + "\n")
with (PROJECT / "requirements-runtime.txt").open("w") as stream:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
runtime = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "numpy": importlib.metadata.version("numpy"),
    "pandas": importlib.metadata.version("pandas"),
    "pycocotools": importlib.metadata.version("pycocotools"),
}
(PROJECT / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
print("Exact runtime:", json.dumps(runtime), flush=True)

annotation_files = list(Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
assert len(annotation_files) == 1, f"Expected one official annotation file, found {len(annotation_files)}"
DATA_ROOT = annotation_files[0].parent.parent
print("Official competition data:", DATA_ROOT, flush=True)

config_path = PROJECT / "configs/neural.json"
config = json.loads(config_path.read_text())
config["device"] = "cuda"
runtime_config_path = PROJECT / "configs/neural-kaggle.json"
runtime_config_path.write_text(json.dumps(config, indent=2) + "\n")
(PROJECT / "submissions").mkdir()

def cli(command, *arguments):
    args = [
        sys.executable, "-u", "-m", "solarseg.cli", command,
        "--data-root", str(DATA_ROOT), *map(str, arguments),
    ]
    print("RUN:", " ".join(args), flush=True)
    subprocess.run(args, cwd=PROJECT, check=True)

cli("prepare", "--output", "data/processed")
cli("classical-cv", "--config", "configs/candidates.json", "--output", "runs/classical_v1")

# The first valid submission is based entirely on development-selected classical
# parameters, before neural training or any locked-holdout evaluation.
selection = json.loads(Path("runs/classical_v1/selection.json").read_text())
candidate = dict(selection["candidate"], mode="classical")
baseline = Path("runs/baseline_v1")
baseline.mkdir()
(baseline / "model.json").write_text(json.dumps({
    "candidate": candidate,
    "development_pq": selection["development_pq"],
    "selected_by": "classical development PQ before neural CV; locked holdout unused",
}, indent=2) + "\n")
cli("infer", "--model", baseline / "model.json", "--device", "cuda", "--output", "submissions/baseline_v1.csv")

cli("neural-cv", "--config", runtime_config_path, "--output", "runs/unet_v1")
cli("finalize", "--classical-run", "runs/classical_v1", "--neural-run", "runs/unet_v1", "--output", "runs/final_v1")
cli("infer", "--model", "runs/final_v1/model.json", "--device", "cuda", "--output", "submissions/final_v1.csv")

# Archive reproducible source, configurations, weights, predictions and evidence.
# Training polygons/consensus masks and original competition images stay outside
# this export. The original requirements and the exact Kaggle freeze both ship.
output_archive = Path("/kaggle/working/solar_filament_2026_outputs.tar.gz")
with tarfile.open(output_archive, "w:gz") as archive:
    for relative in (
        "src", "configs", "requirements.txt", "requirements-runtime.txt",
        "hardware.json", "runtime.json", "source-manifest.json", "runs", "submissions",
        "data/processed/data_audit.json", "data/processed/train_manifest.csv",
        "data/processed/test_manifest.csv",
    ):
        archive.add(PROJECT / relative, arcname=str(Path("project") / relative))
print("Completed output archive:", output_archive, flush=True)
print("Competition CSV files are generated locally in Kaggle; no leaderboard upload was performed.", flush=True)
'''


def build(
    owner: str, slug: str, output_dir: Path, title: str,
    machine_shape: str = "Gpu", expected_gpu: str = "T4",
) -> dict:
    project = Path(__file__).resolve().parents[1]
    source_files = sorted((project / "src/solarseg").glob("*.py")) + [
        project / "configs/candidates.json",
        project / "configs/neural.json",
        project / "requirements.txt",
    ]
    source_manifest = []
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in source_files:
            relative = path.relative_to(project).as_posix()
            content = path.read_bytes()
            entry = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content)
            source_manifest.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})
    payload = stream.getvalue()
    source = textwrap.dedent(KERNEL_TEMPLATE).lstrip().replace(
        "__OWNER_JSON__", json.dumps(owner)
    ).replace(
        "__SLUG_JSON__", json.dumps(slug)
    ).replace(
        "__MACHINE_SHAPE_JSON__", json.dumps(machine_shape)
    ).replace(
        "__EXPECTED_GPU_JSON__", json.dumps(expected_gpu)
    ).replace(
        "__ZIP_SHA_JSON__", json.dumps(hashlib.sha256(payload).hexdigest())
    ).replace(
        "__BUNDLE_JSON__", json.dumps(base64.b64encode(payload).decode("ascii"))
    ).replace(
        "__MANIFEST_JSON__", repr(source_manifest)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    code_file = "solar_filament_pipeline.py"
    (output_dir / code_file).write_text(source)
    metadata = {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": machine_shape,
        "session_timeout_seconds": 43200,
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": [f"{owner}/solar-filament-runtime-v1"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "kernel": metadata["id"], "directory": str(output_dir),
        "code_file": code_file, "source_zip_sha256": hashlib.sha256(payload).hexdigest(),
        "source_files": len(source_files), "status": "built_not_submitted",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="mengxinlab")
    parser.add_argument("--slug", default="solar-filament-2026-offline-20261030")
    parser.add_argument("--title", default="Solar Filament 2026 Offline 20261030")
    parser.add_argument("--machine-shape", default="Gpu")
    parser.add_argument("--expected-gpu", default="T4")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path("outputs/kaggle") / f"{args.owner}_{args.slug}"
    print(json.dumps(build(
        args.owner, args.slug, output, args.title,
        args.machine_shape, args.expected_gpu,
    ), indent=2))


if __name__ == "__main__":
    main()
