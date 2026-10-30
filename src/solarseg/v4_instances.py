"""V4 instance formation and reporting from existing official PQ counts.

``instances`` accepts a full-resolution float32 probability map. A value of -1
marks pixels outside the solar disk; an explicit Boolean disk may additionally
restrict it. Both modes threshold and close the mask, discard connected
components smaller than ``seed_min_area``, and apply the final instance-area
limits. ``cc`` retains the remaining components separately. ``group`` labels
the union of their radius-``group_radius`` dilations, then maps those group IDs
back to the original foreground. Dilation changes membership, never boundaries.
Small retained fragments can therefore pass the final area cut as one group.

Required config keys and frozen V4 values:
    mode: "cc" or "group"
    threshold: 0.45
    closing_radius: 1
    seed_min_area: 16
    group_radius: 6  # elliptical footprint; discrete connectivity is 8-neighbour
    min_area: 400
    max_area: 100000

No elongation filter is used (equivalent to v3 min_elongation=1). The grouping
idea is informed by Amey Pawar's MIT-licensed filament-seg, commit
81a739bb7786a44b989f5d76afbb6f9d22b5d1dd, postprocess.py:110-124:
https://github.com/ameyypawar/filament-seg
This implementation is newly written for the stated V4 contract; it imports no
code, weights or detector package from that project.

``summarize`` only pools already-scored annotator-image records. It does not
rerun matching or replace ``solarseg.metrics.score_frames``. The latter retains
the v3 float64 COCO-IoU arithmetic; it is a numerical approximation to the
official notebook's float32 overlap arithmetic, as documented by the P1 audit.
"""

import cv2
import numpy as np
import pandas as pd


def instances(
    probability: np.ndarray,
    disk: np.ndarray | None,
    config: dict,
) -> list[np.ndarray]:
    """Form disjoint uint8 instances without adding dilation pixels to masks."""
    binary = ((probability > config["threshold"]) & (probability >= 0)).astype(np.uint8)
    if disk is not None:
        binary &= disk.astype(np.uint8)
    radius = config["closing_radius"]
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    )
    binary &= (probability >= 0).astype(np.uint8)
    if disk is not None:
        binary &= disk.astype(np.uint8)

    _, component_labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    retain = stats[:, cv2.CC_STAT_AREA] >= config["seed_min_area"]
    retain[0] = False
    foreground = retain[component_labels].astype(np.uint8)

    if config["mode"] == "cc":
        group_labels = np.where(foreground > 0, component_labels, 0)
    elif config["mode"] == "group":
        radius = config["group_radius"]
        expanded = cv2.dilate(
            foreground,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
        )
        _, expanded_labels = cv2.connectedComponents(expanded, connectivity=8)
        group_labels = np.where(foreground > 0, expanded_labels, 0)
    else:
        raise ValueError(f"Unknown V4 instance mode: {config['mode']}")

    areas = np.bincount(group_labels.ravel())
    keep = (areas >= config["min_area"]) & (areas <= config["max_area"])
    keep[0] = False
    return [(group_labels == label).astype(np.uint8) for label in np.flatnonzero(keep)]


def _pooled(rows: pd.DataFrame) -> dict:
    totals = rows[["tp", "fp", "fn", "sum_iou"]].sum()
    tp, fp, fn = (int(totals[key]) for key in ("tp", "fp", "fn"))
    sum_iou = float(totals["sum_iou"])
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": sum_iou / denominator if denominator else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sum_iou": sum_iou,
        "observations": int(rows["image_id"].nunique()),
        "annotator_image_records": len(rows),
    }


def summarize(per_image_scores: list[dict] | pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Pool existing per-annotator-image counts globally, by fold, and by year.

    The score rows contain ``image_id,tp,fp,fn,sum_iou``; additional fields from
    ``score_frames(...)["per_image"]`` are preserved without affecting pooling.
    Manifest rows contain a unique ``image_id``, integer ``fold`` and ISO
    ``timestamp``. All annotations of an observation inherit that observation's
    metadata. No per-image PQ average is used.
    """
    records = pd.DataFrame(per_image_scores)
    metadata = manifest.set_index("image_id").loc[records["image_id"], ["fold", "timestamp"]]
    records["fold"] = metadata["fold"].to_numpy()
    records["timestamp"] = metadata["timestamp"].to_numpy()
    records["year"] = pd.to_datetime(records["timestamp"]).dt.year
    return {
        "global": _pooled(records),
        "per_fold": {str(int(fold)): _pooled(rows) for fold, rows in records.groupby("fold")},
        "per_year": {str(int(year)): _pooled(rows) for year, rows in records.groupby("year")},
        "metric": "v3 score_frames counts pooled; strict IoU > 0.5 per annotator-image",
        "numerical_note": "float64 COCO IoU; official float32 overlap can differ at threshold rounding",
    }
