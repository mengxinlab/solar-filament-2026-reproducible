"""Official-label preparation, temporal partitions and a single data audit.

Only the competition training segmentation polygons supervise masks. Observation
timestamps define the partitions; they are never model inputs.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util


def _records(manifest) -> list[dict]:
    if isinstance(manifest, pd.DataFrame):
        return manifest.to_dict("records")
    return manifest


def _dates(rows: list[dict]) -> dict[str, datetime]:
    return {
        row["image_id"]: datetime.fromisoformat(row["timestamp"])
        for row in rows
    }


def _outside_embargo(
    candidates: list[dict], validation: list[dict], embargo_days: int
) -> list[dict]:
    dates = _dates(candidates + validation)
    validation_dates = [dates[row["image_id"]] for row in validation]
    embargo = timedelta(days=embargo_days)
    return [
        row for row in candidates
        if all(abs(dates[row["image_id"]] - date) > embargo for date in validation_dates)
    ]


def development_ids(manifest, embargo_days: int = 27) -> list[str]:
    """Development observations excluding the locked-holdout time embargo."""
    rows = _records(manifest)
    development = [row for row in rows if row["fold"] != 3]
    holdout = [row for row in rows if row["fold"] == 3]
    return [
        row["image_id"]
        for row in _outside_embargo(development, holdout, embargo_days)
    ]


def training_ids(
    manifest,
    validation_fold: int,
    development_only: bool = True,
    embargo_days: int = 27,
) -> list[str]:
    """Return temporal-purged training IDs for a development or holdout fold.

    Development CV first excludes the locked holdout and its near neighbours,
    then excludes the selected validation observations and their neighbours.
    With validation_fold=3 this returns all usable development observations.
    """
    rows = _records(manifest)
    if development_only:
        dev_ids = set(development_ids(rows, embargo_days))
        candidates = [row for row in rows if row["image_id"] in dev_ids]
        validation = [
            row for row in rows
            if row["fold"] == validation_fold
            and (validation_fold == 3 or row["image_id"] in dev_ids)
        ]
    else:
        candidates = rows
        validation = [row for row in rows if row["fold"] == validation_fold]
    candidates = [row for row in candidates if row["fold"] != validation_fold]
    return [
        row["image_id"]
        for row in _outside_embargo(candidates, validation, embargo_days)
    ]


def _fold(year: int) -> tuple[int, str]:
    if year <= 2013:
        return 0, "years_2011_2013"
    if year <= 2016:
        return 1, "years_2014_2016"
    if year <= 2019:
        return 2, "years_2017_2019"
    return 3, "years_2020_2022"


def prepare(data_root: Path, output_dir: Path) -> dict:
    """Create manifests, independent-annotator truth and 512² consensus masks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    annotation_path = (
        data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    annotations = json.loads(annotation_path.read_text())
    records_by_file = defaultdict(list)
    annotations_by_record = defaultdict(list)
    for record in annotations["images"]:
        records_by_file[record["file_name"]].append(record)
    for annotation in annotations["annotations"]:
        annotations_by_record[annotation["image_id"]].append(annotation)

    train_files = sorted((data_root / "train" / "train_images").glob("*.jpeg"))
    test_files = sorted((data_root / "test" / "test_images").glob("*.jpeg"))
    train_rows = []
    truth_rows = []
    annotators = set()
    for path in train_files:
        image_id = path.stem
        timestamp = datetime.strptime(image_id[:14], "%Y%m%d%H%M%S")
        fold, group = _fold(timestamp.year)
        train_rows.append({
            "image_id": image_id,
            "file_name": path.name,
            "fold": fold,
            "group": group,
            "timestamp": timestamp.isoformat(),
        })
        records = records_by_file[path.name]
        consensus = np.zeros((512, 512), dtype=np.float32)
        for record in records:
            annotators.add(record["id"].split("-", 1)[0])
            instance_rles = []
            for annotation in annotations_by_record[record["id"]]:
                rle = mask_util.merge(mask_util.frPyObjects(
                    annotation["segmentation"], record["height"], record["width"]
                ))
                instance_rles.append(rle)
                truth_rows.append({
                    "filament_id": f'{record["id"]}_{annotation["id"]}',
                    "segmentation_rle": rle["counts"].decode("ascii"),
                })
            annotator_union = mask_util.decode(mask_util.merge(instance_rles))
            consensus += cv2.resize(
                annotator_union.astype(np.float32),
                (512, 512),
                interpolation=cv2.INTER_AREA,
            )
        consensus /= len(records)
        cv2.imwrite(str(mask_dir / f"{image_id}.png"), np.rint(consensus * 255).astype(np.uint8))

    test_rows = [{"image_id": path.stem, "file_name": path.name} for path in test_files]
    pd.DataFrame(train_rows).to_csv(output_dir / "train_manifest.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "test_manifest.csv", index=False)
    pd.DataFrame(truth_rows).to_csv(output_dir / "truth.csv", index=False)

    # Read image bytes once for exact provenance, dimensions and low-resolution
    # perceptual evidence. Test images are unlabeled audit inputs only.
    sha_groups = defaultdict(list)
    descriptions = []
    thumbnails = []
    shapes = Counter()
    for split, paths in (("train", train_files), ("test", test_files)):
        for path in paths:
            image_bytes = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            shapes[f"{image.shape[0]}x{image.shape[1]}"] += 1
            digest = hashlib.sha256(image_bytes).hexdigest()
            sha_groups[digest].append({"split": split, "image_id": path.stem})
            descriptions.append({"split": split, "image_id": path.stem})
            thumbnails.append(cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).reshape(-1) / 255.)

    thumb = np.asarray(thumbnails, dtype=np.float64)
    centered = thumb - thumb.mean(axis=1, keepdims=True)
    normalized = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    correlation = normalized @ normalized.T
    squared = np.square(thumb).mean(axis=1)
    mse = np.maximum(squared[:, None] + squared[None, :] - 2 * (thumb @ thumb.T) / 1024, 0)
    # A heuristic flag, not proof of duplicate identity or complete leakage absence.
    near = np.triu((correlation >= 0.9995) & (mse <= (2 / 255) ** 2), k=1)
    fold_by_id = {row["image_id"]: row["fold"] for row in train_rows}
    perceptual_pairs = []
    for i, j in zip(*np.nonzero(near)):
        left, right = descriptions[i], descriptions[j]
        cross_split = left["split"] != right["split"]
        cross_fold = (
            left["split"] == right["split"] == "train"
            and fold_by_id[left["image_id"]] != fold_by_id[right["image_id"]]
        )
        if cross_split or cross_fold:
            perceptual_pairs.append({
                "left": left, "right": right,
                "correlation": float(correlation[i, j]),
                "rmse_0_1": float(np.sqrt(mse[i, j])),
            })

    dates = _dates(train_rows)
    ordered = sorted(train_rows, key=lambda row: dates[row["image_id"]])
    chains = [[]]
    for row in ordered:
        if chains[-1] and dates[row["image_id"]] - dates[chains[-1][-1]] > timedelta(days=27):
            chains.append([])
        chains[-1].append(row["image_id"])
    temporal_cross_fold_pairs = []
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            delta = dates[right["image_id"]] - dates[left["image_id"]]
            if delta > timedelta(days=27):
                break
            if left["fold"] != right["fold"]:
                temporal_cross_fold_pairs.append({
                    "left": left["image_id"], "right": right["image_id"],
                    "days_apart": delta.total_seconds() / 86400,
                })

    development = set(development_ids(train_rows))
    per_fold = {}
    for fold in range(4):
        validation = [
            row["image_id"] for row in train_rows
            if row["fold"] == fold and (fold == 3 or row["image_id"] in development)
        ]
        train = training_ids(train_rows, fold)
        per_fold[str(fold)] = {"train_ids": train, "validation_ids": validation}

    def split_summary(paths):
        names = [path.stem for path in paths]
        return {
            "images": len(names),
            "years": dict(sorted(Counter(name[:4] for name in names).items())),
            "stations": dict(sorted(Counter(name[14:] for name in names).items())),
            "timestamp_min": min(name[:14] for name in names),
            "timestamp_max": max(name[:14] for name in names),
        }

    audit = {
        "train": split_summary(train_files),
        "test": split_summary(test_files),
        "annotation_image_records": len(annotations["images"]),
        "annotations": len(annotations["annotations"]),
        "annotators": sorted(annotators),
        "annotator_count": len(annotators),
        "annotation_sets_per_source_image": dict(sorted(Counter(len(value) for value in records_by_file.values()).items())),
        "image_shapes": dict(shapes),
        "exact_duplicates": [
            {"sha256": digest, "images": group}
            for digest, group in sha_groups.items() if len(group) > 1
        ],
        "perceptual_method": {
            "thumbnail": [32, 32], "correlation_min": 0.9995,
            "rmse_max_0_1": 2 / 255,
            "scope": "train-test and cross-fold train pairs",
            "limitation": "Low-resolution similarity is a risk flag; it does not establish labels or prove absence of leakage.",
        },
        "perceptual_duplicate_risk_pairs": perceptual_pairs,
        "temporal_split": {
            "gap_chain_threshold_days": 27,
            "observed_gap_chain_group_count": len(chains),
            "gap_chain_sizes_descending": sorted(map(len, chains), reverse=True),
            "strategy": "Four contiguous year blocks; fold 3 locked; training and development validation exclude a 27-day embargo around the locked holdout; each fold training also excludes a 27-day embargo around its validation.",
            "reason": "The largest 27-day transitive chain contains 576/707 observations and cannot yield balanced independent four-fold partitions.",
            "fold_counts_before_embargo": dict(sorted(Counter(row["fold"] for row in train_rows).items())),
            "cross_fold_pairs_within_27_days_before_embargo": temporal_cross_fold_pairs,
            "development_ids": sorted(development),
            "development_excluded_by_holdout_embargo": [
                row["image_id"] for row in train_rows
                if row["fold"] != 3 and row["image_id"] not in development
            ],
            "per_fold": per_fold,
        },
        "training_target": "512x512 soft consensus: COCO polygon union within each annotator-image, area resize, equal average across independent annotators, uint8 round(probability*255).",
        "label_provenance": str(annotation_path),
        "features_excluded": ["annotation spine", "annotation bbox", "annotation category", "external MAGFiLO ground truth", "test labels", "timestamp and station metadata"],
    }
    (output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
