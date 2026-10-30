"""Audit a frozen CSV against official mounted test IDs; never upload."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pycocotools import mask as coco

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--csv", type=Path, default=Path("submissions/final_v1.csv"))
parser.add_argument("--record", type=Path, required=True)
args = parser.parse_args()
digest = hashlib.sha256(args.csv.read_bytes()).hexdigest()
assert digest == "c8e6ac199a3fbe429497d36d0be1973026aad093e1932ba65f306a7fbdc050f0"
test_ids = {path.stem for path in (args.data_root / "test/test_images").glob("*.jpeg")}
assert len(test_ids) == 180
frame = pd.read_csv(args.csv, keep_default_na=False)
assert frame.columns.tolist() == ["filament_id", "segmentation_rle"]
assert frame.filament_id.is_unique
counts = dict.fromkeys(test_ids, 0)
groups = frame.filament_id.str.rsplit("_", n=1).str[0]
for image_id, rows in frame.groupby(groups):
    assert image_id in test_ids
    occupied = np.zeros((2048, 2048), dtype=np.uint8)
    for row in rows.itertuples(index=False):
        _, suffix = row.filament_id.rsplit("_", 1)
        assert suffix.isdigit() and int(suffix) > 0
        rle = {"size": [2048, 2048], "counts": row.segmentation_rle.encode("ascii")}
        mask = coco.decode(rle)
        assert mask.shape == (2048, 2048) and mask.any()
        assert not np.any(mask & occupied)
        occupied |= mask
        assert coco.encode(np.asfortranarray(mask))["counts"] == rle["counts"]
        counts[image_id] += 1
record = {"csv": str(args.csv), "sha256": digest, "rows": len(frame),
          "test_images_accounted_for": len(test_ids), "images_with_instances": sum(value > 0 for value in counts.values()),
          "zero_instance_images": sorted(key for key, value in counts.items() if value == 0),
          "all_nonempty_disjoint_canonical_2048_rles": True,
          "zero_instance_convention": "no instance row; no dummy mask",
          "official_acceptance_reference": "docs/official_submission_v3.md; no upload performed"}
args.record.parent.mkdir(parents=True, exist_ok=True)
with args.record.open("x") as stream:
    stream.write(json.dumps(record, indent=2)+"\n")
print(json.dumps(record, indent=2))
