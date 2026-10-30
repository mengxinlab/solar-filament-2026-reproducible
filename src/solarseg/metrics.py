"""Official filament PQ, accumulated across annotator–image groups.

Source: docs/reference/official_evaluation/self-evaluation-notebook.ipynb,
cells 3–4. Every pair with IoU strictly greater than 0.5 is a hit; the official
implementation does not use one-to-one assignment. Predictions are compared
separately against every annotator of their image. Only groups represented
in the ground-truth CSV are scored, as in the supplied notebook.
"""

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from .rle import as_rle


def _pq(tp: int, fp: int, fn: int, sum_iou: float) -> float:
    denominator = tp + 0.5 * fp + 0.5 * fn
    return sum_iou / denominator if denominator > 0 else 0.0


def score_frames(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    image_ids: list[str] | None = None,
) -> dict:
    """Score official-format dataframes, optionally restricted to image IDs.

    Both frames contain ``filament_id`` and ``segmentation_rle``. Truth IDs
    start with ``<annotator>-<image_id>_``; prediction IDs start with
    ``<image_id>_``. Empty prediction frames must retain these columns.
    ``pq`` is the ratio of accumulated counts, not the mean of image PQs.
    """
    truth_groups = truth["filament_id"].str.split("_", n=1).str[0]
    prediction_images = predictions["filament_id"].str.split("_", n=1).str[0]

    if image_ids is not None:
        truth_images = truth_groups.str.split("-", n=1).str[1]
        selection = truth_images.isin(image_ids)
        truth = truth.loc[selection]
        truth_groups = truth_groups.loc[selection]
        selection = prediction_images.isin(image_ids)
        predictions = predictions.loc[selection]
        prediction_images = prediction_images.loc[selection]

    predictions_by_image = {
        image_id: rows["segmentation_rle"].tolist()
        for image_id, rows in predictions.groupby(prediction_images, sort=False)
    }
    per_image = []
    tp_total = fp_total = fn_total = 0
    sum_iou_total = 0.0

    for annotator_image, rows in truth.groupby(truth_groups, sort=False):
        annotator, image_id = annotator_image.split("-", 1)
        gt_rles = [as_rle(counts) for counts in rows["segmentation_rle"]]
        pred_rles = [
            as_rle(counts) for counts in predictions_by_image.get(image_id, [])
        ]
        n_gt, n_pred = len(gt_rles), len(pred_rles)
        if n_gt == 0 or n_pred == 0:
            iou = np.zeros((n_gt, n_pred), dtype=np.float64)
        else:
            iou = mask_util.iou(pred_rles, gt_rles, [0] * n_gt).T

        hits = iou > 0.5
        tp = int(hits.sum())
        fp = int((hits.sum(axis=0) == 0).sum())
        fn = int((hits.sum(axis=1) == 0).sum())
        sum_iou = float(iou[hits].sum())
        per_image.append(
            {
                "annotator": annotator,
                "image_id": image_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "sum_iou": sum_iou,
                "pq": _pq(tp, fp, fn, sum_iou),
            }
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn
        sum_iou_total += sum_iou

    return {
        "pq": _pq(tp_total, fp_total, fn_total, sum_iou_total),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "sum_iou": sum_iou_total,
        "per_image": per_image,
    }
