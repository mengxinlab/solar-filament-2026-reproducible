"""Create an offline V4-only Kaggle source snapshot; never submit or train."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile


TEMPLATE = r'''
"""Prespecified Solar V4 development experiments, isolated from frozen V3."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"

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
SOURCE_ZIP_SHA256 = __ZIP_SHA_JSON__
SOURCE_MANIFEST = __MANIFEST_JSON__
SOURCE_B64 = __SOURCE_B64_JSON__

wheel_files = list(Path("/kaggle/input").rglob("pycocotools-2.0.10-*.whl"))
assert len(wheel_files) == 1, f"Expected one offline pycocotools wheel, found {len(wheel_files)}"
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(wheel_files[0])
], check=True)

import torch
torch.set_num_threads(4)
assert torch.cuda.is_available(), "V4 requires the ordinary Kaggle CUDA GPU"
properties = torch.cuda.get_device_properties(0)
capability = torch.cuda.get_device_capability(0)
print("CUDA device:", properties.name, flush=True)
print("CUDA capability:", capability, flush=True)
print("VRAM GB:", round(properties.total_memory / 1024**3, 1), flush=True)
assert "T4" in properties.name.upper(), f"Expected T4, got {properties.name}"

PROJECT = Path("/kaggle/working/project")
PROJECT.mkdir()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as source_archive:
    source_archive.extractall(PROJECT)
os.chdir(PROJECT)
os.environ["PYTHONPATH"] = str(PROJECT / "src")

hardware = {
    "name": properties.name,
    "capability": list(capability),
    "vram_bytes": properties.total_memory,
    "cuda_available": True,
    "torch_threads": torch.get_num_threads(),
    "requested_machine_shape": "Gpu",
    "kernel": f"{OWNER}/{SLUG}",
    "session_timeout_seconds": 7200,
}
(PROJECT / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
runtime = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "numpy": importlib.metadata.version("numpy"),
    "pandas": importlib.metadata.version("pandas"),
    "pycocotools": importlib.metadata.version("pycocotools"),
}
(PROJECT / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
with (PROJECT / "requirements-runtime.txt").open("w") as stream:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
(PROJECT / "source-manifest.json").write_text(json.dumps({
    "source_zip_sha256": SOURCE_ZIP_SHA256,
    "files": SOURCE_MANIFEST,
}, indent=2) + "\n")
print("Exact runtime:", json.dumps(runtime), flush=True)

annotation_files = list(Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
assert len(annotation_files) == 1, f"Expected one official Solar train JSON, found {len(annotation_files)}"
data_root = annotation_files[0].parent.parent
v3_root = PROJECT / "evidence/v3"
print("Official Solar data:", data_root, flush=True)
print("Read-only frozen V3 source:", v3_root, flush=True)

command = [
    sys.executable, "-u", "-m", "solarseg.v4_runner",
    "--data-root", str(data_root),
    "--v3-root", str(v3_root),
    "--config", "configs/v4_experiments.json",
    "--output", "runs/v4_native",
]
print("RUN V4 ONLY:", " ".join(command), flush=True)
with (PROJECT / "execution.log").open("w") as log:
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, cwd=PROJECT) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

def export_filter(info):
    # Raw development images and rendered targets remain within the Kaggle
    # session, outside this reproducibility export. Summary JSON is retained.
    if Path(info.name).parts[:4] == ("project", "runs", "v4_native", "cache"):
        return None
    return info

output = Path("/kaggle/working/solar_v4_outputs.tar.gz")
with tarfile.open(output, "w:gz") as archive:
    for relative in (
        "src", "configs", "requirements.txt", "requirements-runtime.txt",
        "hardware.json", "runtime.json", "source-manifest.json", "execution.log",
        "runs/v4_native",
    ):
        archive.add(PROJECT / relative, arcname=str(Path("project") / relative), filter=export_filter)
print("Completed V4 output archive:", output, flush=True)
print("No V3 run or holdout evaluation was repeated. No leaderboard upload was performed.", flush=True)
'''


def build(owner: str, slug: str, title: str, output_dir: Path) -> dict:
    project = Path(__file__).resolve().parents[1]
    files = sorted((project / "src/solarseg").glob("*.py")) + [
        project / "configs/v4_experiments.json",
        project / "configs/v4_splits.json",
        project / "requirements.txt",
        project / "evidence/v3/data/processed/train_manifest.csv",
        project / "evidence/v3/runs/classical_v1/scores.json",
        project / "evidence/v3/runs/classical_v1/public_reference_sigma64_per_image.csv",
    ]
    manifest = []
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(project).as_posix()
            content = path.read_bytes()
            entry = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content)
            manifest.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})
    payload = stream.getvalue()
    replacements = {
        "__OWNER_JSON__": json.dumps(owner),
        "__SLUG_JSON__": json.dumps(slug),
        "__ZIP_SHA_JSON__": json.dumps(hashlib.sha256(payload).hexdigest()),
        "__MANIFEST_JSON__": repr(manifest),
        "__SOURCE_B64_JSON__": json.dumps(base64.b64encode(payload).decode("ascii")),
    }
    source = textwrap.dedent(TEMPLATE).lstrip()
    for token, value in replacements.items():
        source = source.replace(token, value)
    output_dir.mkdir(parents=True, exist_ok=False)
    code_file = "solar_v4_pipeline.py"
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
        "machine_shape": "Gpu",
        "session_timeout_seconds": 7200,
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": [f"{owner}/solar-filament-runtime-v1"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = {
        "kernel": metadata["id"],
        "directory": str(output_dir),
        "code_file": code_file,
        "source_zip_sha256": hashlib.sha256(payload).hexdigest(),
        "source_files": len(files),
        "status": "built_not_submitted",
        "session_timeout_seconds": 7200,
    }
    (output_dir / "build-record.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="mengxinlab")
    parser.add_argument("--slug", default="solar-filament-v4-native-20261030")
    parser.add_argument("--title", default="Solar Filament V4 Native 20261030")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kaggle/mengxinlab_solar_v4_native"))
    args = parser.parse_args()
    print(json.dumps(build(args.owner, args.slug, args.title, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
