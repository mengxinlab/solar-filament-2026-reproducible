"""Prespecified native-resolution experiments; never access frozen holdout labels."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import torch

from .metrics import score_frames
from .pipeline import COLUMNS, code_hash, encode_instances
from .rle import decode
from .v4_instances import instances, summarize
from .v4_model import prepare_cache, train_variant, load_variant, predict_full


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str)+"\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/v4_experiments.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    splits = json.loads(Path("configs/v4_splits.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "prespecified_config.json", config)
    save(args.output / "provenance.json", {"source_sha256": code_hash(),
         "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
         "split_sha256": hashlib.sha256(Path("configs/v4_splits.json").read_bytes()).hexdigest(),
         "reference_artifact": str(args.v3_root), "holdout_labels_used": False,
         "test_images_used": False, "public_score_used_for_selection": False})
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    started = perf_counter()
    manifest = pd.read_csv(args.v3_root / "data/processed/train_manifest.csv")
    development = manifest.set_index("image_id").loc[splits["development_ids"]].reset_index()
    assert set(development.fold) == {0, 1, 2}
    cache = args.output / "cache"
    save(args.output / "cache_summary.json", prepare_cache(args.data_root, development,
         splits["development_ids"], cache))
    truth = pd.read_csv(cache / "dev_truth.csv")
    baseline_scores = json.loads((args.v3_root / "runs/classical_v1/scores.json").read_text())["public_reference_sigma64"]
    baseline_per_image = pd.read_csv(args.v3_root / "runs/classical_v1/public_reference_sigma64_per_image.csv")
    all_results = {}
    compute = []
    for variant in config["variants"]:
        variant_dir = args.output / variant
        variant_dir.mkdir()
        all_predictions = {mode: [] for mode in ("cc", "group")}
        probability_stats = []
        for fold in config["folds"]:
            fold_dir = variant_dir / f"fold{fold}"
            train_ids = splits["folds"][str(fold)]["train_ids"]
            val_ids = splits["folds"][str(fold)]["validation_ids"]
            assert set(train_ids).isdisjoint(val_ids)
            run_config = dict(config, variant=variant, seed=config["seed"]+fold)
            train_started = perf_counter()
            torch.cuda.reset_peak_memory_stats()
            train_variant(development, train_ids, cache, run_config, fold_dir)
            train_seconds = perf_counter()-train_started
            model = load_variant(fold_dir / "model.pt", config["device"])
            inference_started = perf_counter()
            fold_predictions = {mode: [] for mode in all_predictions}
            rows = development.set_index("image_id").loc[val_ids]
            for index, (image_id, row) in enumerate(rows.iterrows(), 1):
                image = cv2.imread(str(cache / "raw" / f"{image_id}.png"), 0)
                p = predict_full(model, image, config["device"], run_config)
                inside = p >= 0
                probability_stats.append({"image_id": image_id, "fold": fold,
                    "max_probability": float(p[inside].max()), "mean_probability": float(p[inside].mean()),
                    "foreground_fraction_at_fixed_threshold": float((p[inside] > config["postprocessing"]["threshold"]).mean())})
                for mode in fold_predictions:
                    post_config = dict(config["postprocessing"], mode=mode)
                    masks = instances(p, None, post_config)
                    fold_predictions[mode].extend(encode_instances(image_id, masks))
                if index % 25 == 0:
                    print(json.dumps({"variant": variant, "fold": fold, "inference_images": index,
                                      "total": len(rows)}), flush=True)
            for mode, records in fold_predictions.items():
                frame = pd.DataFrame(records, columns=COLUMNS)
                frame.to_csv(fold_dir / f"{mode}_oof.csv", index=False)
                all_predictions[mode].append(frame)
            compute.append({"variant": variant, "fold": fold, "train_images": len(train_ids),
                 "validation_images": len(val_ids), "updates": config["epochs"]*config["steps_per_epoch"],
                 "training_seconds": train_seconds, "inference_seconds": perf_counter()-inference_started,
                 "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                 "inference_tiles_per_image": 25, "tta": False})
            save(args.output / "compute.json", compute)
            del model
        pd.DataFrame(probability_stats).to_csv(variant_dir / "probability_summary.csv", index=False)
        for mode, frames in all_predictions.items():
            frame = pd.concat(frames, ignore_index=True)
            frame.to_csv(variant_dir / f"{mode}_oof.csv", index=False)
            score = score_frames(truth, frame, splits["development_ids"])
            per_image = pd.DataFrame(score["per_image"])
            per_image.to_csv(variant_dir / f"{mode}_per_image.csv", index=False)
            summary = summarize(per_image, development)
            all_results[f"{variant}_{mode}"] = summary
            save(variant_dir / f"{mode}_scores.json", summary)
            failure_cases(variant_dir / f"{mode}_failures", per_image, frame, truth, development, cache,
                          baseline_per_image)
        save(args.output / "results.json", all_results)
    best_name = max(all_results, key=lambda name: all_results[name]["global"]["pq"])
    best = all_results[best_name]
    gate = config["continue_gpu_if"]
    periods_improved = sum(best["per_fold"][str(fold)]["pq"] > baseline_scores["folds"][str(fold)]
                           for fold in config["folds"])
    eligible = (best["global"]["pq"] >= gate["pooled_pq_min"]
                and periods_improved >= gate["periods_improved_min"]
                and min(v["pq"] for v in best["per_fold"].values()) > gate["minimum_period_pq"])
    decision = {"candidate": best_name, "best": best, "baseline": baseline_scores,
        "periods_improved": periods_improved, "continue_gpu": eligible,
        "rule": gate, "elapsed_seconds": perf_counter()-started,
        "interpretation": "Budget gate on reused development groups, not independent validation or significance",
        "holdout_policy": "v3 single frozen holdout result retained; V4 does not inspect or score its labels",
        "next_action": "One confirmatory seed/scale test is justified" if eligible else "Stop additional GPU spending; retain official v3 baseline",
        "automatic_further_runs": False}
    save(args.output / "decision.json", decision)
    print(json.dumps(decision, default=str), flush=True)


def failure_cases(output, scores, predictions, truth, manifest, cache, baseline):
    """Six development failure cases, selected by existing counts, no rescoring."""
    output.mkdir()
    image_scores = scores.groupby("image_id")[["tp", "fp", "fn", "sum_iou"]].sum()
    denominator = image_scores.tp + .5*image_scores.fp + .5*image_scores.fn
    image_scores["pq"] = image_scores.sum_iou/denominator
    image_scores["fold"] = manifest.set_index("image_id").loc[image_scores.index, "fold"]
    cases = []
    for fold in (0, 1, 2):
        selected = image_scores[image_scores.fold == fold].sort_values(["fn", "fp"], ascending=False).head(2)
        for image_id, values in selected.iterrows():
            image = cv2.imread(str(cache / "raw" / f"{image_id}.png"), 0)
            panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            pred_rows = predictions[predictions.filament_id.str.startswith(image_id+"_")]
            pred = np.zeros((2048, 2048), np.uint8)
            for counts in pred_rows.segmentation_rle:
                pred |= decode(counts)
            gt_rows = truth[truth.filament_id.str.contains("-"+image_id+"_", regex=False)]
            annotator = gt_rows.iloc[0].filament_id.split("_")[0]
            gt = np.zeros_like(pred)
            for counts in gt_rows[gt_rows.filament_id.str.startswith(annotator+"_")].segmentation_rle:
                gt |= decode(counts)
            for mask, color in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel, contours, -1, color, 2)
            cv2.imwrite(str(output / f"{image_id}.jpg"), cv2.resize(panel, (768, 768)))
            baseline_rows = baseline[baseline.image_id == image_id]
            cases.append({"image_id": image_id, "fold": fold, **values.to_dict(),
                          "shown_annotator": annotator, "gt_contour": "green", "prediction_contour": "red",
                          "baseline_counts": baseline_rows.to_dict("records")})
    save(output / "cases.json", cases)


if __name__ == "__main__":
    main()
