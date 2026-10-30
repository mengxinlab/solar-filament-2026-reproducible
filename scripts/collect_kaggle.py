"""Download a completed kernel's explicit output archive without recomputation."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    api = KaggleApi()
    api.authenticate()
    owner, slug = args.ref.split("/", 1)
    assert api.config_values["username"] == owner
    args.output.mkdir(parents=True, exist_ok=False)
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = owner
    request.kernel_slug = slug
    request.file_path = "solar_filament_2026_outputs.tar.gz"
    with api.build_kaggle_client() as client:
        redirect = client.kernels.kernels_api_client.download_kernel_output(request)
    archive_path = args.output / request.file_path
    with urlopen(redirect.url) as response, archive_path.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    with tarfile.open(archive_path) as archive:
        archive.extractall(args.output, filter="data")
    print(json.dumps({"kernel": args.ref, "archive": str(archive_path),
                      "project": str(args.output / "project")}, indent=2))


if __name__ == "__main__":
    main()
