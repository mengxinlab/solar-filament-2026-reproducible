"""SaveAndRun an explicit snapshot under the currently authenticated account."""
import argparse
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
args = parser.parse_args()
meta = json.loads((args.folder / "kernel-metadata.json").read_text())
api = KaggleApi()
api.authenticate()
assert api.config_values["username"] == meta["id"].split("/")[0]
request = ApiSaveKernelRequest()
request.slug = meta["id"]
request.new_title = meta["title"]
request.text = (args.folder / meta["code_file"]).read_text()
request.language = meta["language"]
request.kernel_type = meta["kernel_type"]
request.is_private = meta["is_private"]
request.enable_gpu = meta["enable_gpu"]
request.enable_tpu = meta["enable_tpu"]
request.enable_internet = meta["enable_internet"]
request.machine_shape = meta["machine_shape"]
request.dataset_data_sources = meta["dataset_sources"]
request.kernel_data_sources = meta["kernel_sources"]
request.competition_data_sources = meta["competition_sources"]
request.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
request.session_timeout_seconds = meta["session_timeout_seconds"]
with api.build_kaggle_client() as client:
    response = client.kernels.kernels_api_client.save_kernel(request)
record = {"ref": response.ref, "version": response.version_number, "url": response.url,
          "error": response.error, "session_timeout_seconds": request.session_timeout_seconds}
(args.folder / "save-and-run-receipt.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record, indent=2))
assert not response.error, response.error
