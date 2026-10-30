"""Fixed candidate evaluation, chronological CV and reproducible inference."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd

from .classical import probability, instances_from_probability
from .metrics import score_frames
from .rle import encode, as_rle


COLUMNS = ["filament_id", "segmentation_rle"]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path("src").rglob("*.py")):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def begin_run(output: Path, config: dict):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provenance.json", {"code_sha256": code_hash(), "config": config})


def encode_instances(image_id, masks):
    return [{"filament_id": f"{image_id}_{i + 1}", "segmentation_rle": encode(mask)}
            for i, mask in enumerate(masks)]


def accumulated(records):
    tp = int(sum(r["tp"] for r in records))
    fp = int(sum(r["fp"] for r in records))
    fn = int(sum(r["fn"] for r in records))
    total = float(sum(r["sum_iou"] for r in records))
    denominator = tp + .5 * fp + .5 * fn
    return {"pq": total / denominator if denominator else 0., "tp": tp, "fp": fp, "fn": fn, "sum_iou": total}


def evaluate_candidates(truth, frames, manifest, ids, output):
    scores = {}
    for name, frame in frames.items():
        result = score_frames(truth, frame, ids)
        pd.DataFrame(result.pop("per_image")).to_csv(output / f"{name}_per_image.csv", index=False)
        result["folds"] = {
            str(fold): score_frames(truth, frame, rows.image_id.tolist())["pq"]
            for fold, rows in manifest[manifest.image_id.isin(ids)].groupby("fold")
        }
        scores[name] = result
    write_json(output / "scores.json", scores)
    return scores


def classical_frames(data_root, manifest, configs, workers):
    def process(row):
        image = cv2.imread(str(data_root / "train" / "train_images" / row.file_name), 0)
        sigmas = sorted({s for c in configs for s in c["sigmas"]})
        maps = {s: probability(image, s) for s in sigmas}
        return {
            c["id"]: encode_instances(row.image_id, instances_from_probability(
                np.mean([maps[s] for s in c["sigmas"]], axis=0), c))
            for c in configs
        }
    records = {c["id"]: [] for c in configs}
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(process, manifest.itertuples(index=False)), 1):
            for key in records:
                records[key].extend(result[key])
            if i % 25 == 0:
                print(f"classical {i}/{len(manifest)} images; {perf_counter()-started:.1f}s", flush=True)
    return {name: pd.DataFrame(rows, columns=COLUMNS) for name, rows in records.items()}


def classical_cv(data_root: Path, processed: Path, candidate_file: Path, output: Path, workers: int):
    from .prepare import development_ids
    configs = json.loads(candidate_file.read_text())
    begin_run(output, {"stage": "classical_cv", "candidates": configs, "workers": workers})
    manifest = pd.read_csv(processed / "train_manifest.csv")
    ids = development_ids(manifest)
    development = manifest[manifest.image_id.isin(ids)]
    frames = classical_frames(data_root, development, configs, workers)
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}_oof.csv", index=False)
    scores = evaluate_candidates(pd.read_csv(processed / "truth.csv"), frames, manifest, ids, output)
    best = max(scores, key=lambda n: scores[n]["pq"])
    chosen = next(c for c in configs if c["id"] == best)
    write_json(output / "selection.json", {"candidate": chosen, "development_pq": scores[best]["pq"],
               "selection_basis": "development PQ only; locked fold 3 unread",
               "reference_caveat": "public_reference has prior public competition tuning; other candidates prespecified"})
    print(json.dumps({"classical_best": best, "development_pq": scores[best]["pq"]}), flush=True)


def neural_configs():
    common = {"sigmas": [64], "sigma": 64, "max_area": 100000, "min_elongation": 1.0}
    return [dict(common, id=f"unet_t{int(t*100)}_c{close}", threshold=t, closing_radius=close, min_area=400,
                 mode="neural") for t, close in [(0.3, 1), (0.45, 1), (0.6, 1), (0.45, 3)]] + [
        dict(common, id="hybrid_50", threshold=.45, closing_radius=1, min_area=400, mode="hybrid")]


def neural_frames(model, rows, data_root, config, candidates, map_dir):
    from .neural import predict_probability
    records = {c["id"]: [] for c in candidates}
    map_dir.mkdir(parents=True)
    for i, row in enumerate(rows.itertuples(index=False), 1):
        image = cv2.imread(str(data_root / "train" / "train_images" / row.file_name), 0)
        p = predict_probability(model, image, config["device"], config["size"], tta=True)
        np.save(map_dir / f"{row.image_id}.npy", cv2.resize(p, (512, 512)).astype(np.float16))
        contrast = probability(image, 64)
        disk = contrast >= 0
        for c in candidates:
            q = (p + np.clip(contrast / .1, 0, 1)) / 2 if c["mode"] == "hybrid" else p.copy()
            q[~disk] = -1
            records[c["id"]].extend(encode_instances(row.image_id, instances_from_probability(q, c)))
        if i % 25 == 0:
            print(f"neural inference {i}/{len(rows)}", flush=True)
    return {n: pd.DataFrame(r, columns=COLUMNS) for n, r in records.items()}


def neural_cv(data_root: Path, processed: Path, config_file: Path, output: Path):
    from .neural import train_model, load_model
    from .prepare import training_ids, development_ids
    config = json.loads(config_file.read_text())
    candidates = neural_configs()
    begin_run(output, {"stage": "neural_cv", "training": config, "candidates": candidates})
    manifest = pd.read_csv(processed / "train_manifest.csv")
    ids = development_ids(manifest)
    frames = {c["id"]: [] for c in candidates}
    for fold in (0, 1, 2):
        train_ids = training_ids(manifest, fold)
        val = manifest[(manifest.fold == fold) & manifest.image_id.isin(ids)]
        print(f"training fold {fold}: {len(train_ids)} train / {len(val)} validation", flush=True)
        fold_dir = output / f"fold{fold}"
        run_config = dict(config, seed=config["seed"] + fold)
        train_model(manifest, train_ids, data_root, processed, run_config, fold_dir)
        model = load_model(fold_dir / "model.pt", config["device"])
        fold_frames = neural_frames(model, val, data_root, config, candidates, fold_dir / "probabilities")
        for name in frames:
            frames[name].append(fold_frames[name])
        del model
    frames = {n: pd.concat(v, ignore_index=True) for n, v in frames.items()}
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}_oof.csv", index=False)
    scores = evaluate_candidates(pd.read_csv(processed / "truth.csv"), frames, manifest, ids, output)
    best = max(scores, key=lambda n: scores[n]["pq"])
    write_json(output / "selection.json", {"candidate": next(c for c in candidates if c["id"] == best),
               "development_pq": scores[best]["pq"], "training": config})
    print(json.dumps({"neural_best": best, "development_pq": scores[best]["pq"]}), flush=True)


def make_submission(data_root: Path, model_file: Path, output: Path, device: str):
    from .neural import load_model, predict_probability
    model_spec = json.loads(model_file.read_text())
    c = model_spec["candidate"]
    if c["mode"] in ("neural", "hybrid"):
        model = load_model(Path(model_spec["checkpoint"]), device)
    rows = []
    summary = []
    started = perf_counter()
    for path in sorted((data_root / "test" / "test_images").glob("*.jpeg")):
        image = cv2.imread(str(path), 0)
        if c["mode"] == "classical":
            p = np.mean([probability(image, s) for s in c["sigmas"]], axis=0)
        else:
            p = predict_probability(model, image, device, model_spec["training"]["size"], tta=True)
            contrast = probability(image, 64)
            if c["mode"] == "hybrid":
                p = (p + np.clip(contrast / .1, 0, 1)) / 2
            p[contrast < 0] = -1
        masks = instances_from_probability(p, c)
        rows.extend(encode_instances(path.stem, masks))
        summary.append({"image_id": path.stem, "instances": len(masks)})
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.to_csv(output, index=False)
    # One file-level contract validation after generation, including roundtrip
    # CSV reading. No empty dummy instances and no forced minimum detection count.
    validation = validate_submission(output, [r["image_id"] for r in summary])
    record = {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "model": str(model_file), "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
              "seconds": perf_counter()-started, "validation": validation,
              "images": summary, "kaggle_status": "not_uploaded", "public_score": None,
              "private_score": None}
    write_json(output.with_suffix(".json"), record)
    with (output.parent / "ledger.jsonl").open("a") as stream:
        stream.write(json.dumps({k:v for k,v in record.items() if k != "images"})+"\n")
    print(json.dumps({k:v for k,v in record.items() if k != "images"}), flush=True)


def validate_submission(path: Path, image_ids: list[str]):
    from pycocotools import mask as mask_util
    frame = pd.read_csv(path, keep_default_na=False)
    assert frame.columns.tolist() == COLUMNS
    assert frame.filament_id.is_unique
    found_images = set()
    total_area = 0
    for row in frame.itertuples(index=False):
        image_id, suffix = row.filament_id.rsplit("_", 1)
        assert image_id in image_ids and suffix
        rle = as_rle(row.segmentation_rle)
        mask = mask_util.decode(rle)
        assert mask.shape == (2048, 2048)
        area = int(mask_util.area(rle))
        assert area > 0
        found_images.add(image_id)
        total_area += area
    return {"rows": len(frame), "images": len(found_images), "total_area": total_area,
            "images_without_detections": sorted(set(image_ids)-found_images),
            "format": "2048x2048 COCO compressed ASCII RLE", "status": "local_contract_pass"}


def finalize(data_root: Path, processed: Path, classical_dir: Path, neural_dir: Path, output: Path):
    from .neural import train_model, load_model
    from .prepare import training_ids
    a = json.loads((classical_dir / "selection.json").read_text())
    b = json.loads((neural_dir / "selection.json").read_text())
    selected = a if a["development_pq"] >= b["development_pq"] else b
    candidate = dict(selected["candidate"])
    candidate["mode"] = "classical" if selected is a else candidate["mode"]
    begin_run(output, {"stage": "locked_holdout_then_full_fit", "selected_before_holdout": selected,
                       "classical_development_pq": a["development_pq"], "neural_development_pq": b["development_pq"]})
    write_json(output / "selected_before_holdout.json", selected)
    manifest = pd.read_csv(processed / "train_manifest.csv")
    holdout = manifest[manifest.fold == 3]
    if candidate["mode"] == "classical":
        frame = classical_frames(data_root, holdout, [candidate], 4)[candidate["id"]]
    else:
        train_model(manifest, training_ids(manifest, 3), data_root, processed, b["training"], output / "holdout_model")
        model = load_model(output / "holdout_model/model.pt", b["training"]["device"])
        frame = neural_frames(model, holdout, data_root, b["training"], [candidate], output / "holdout_maps")[candidate["id"]]
        del model
    frame.to_csv(output / "holdout_predictions.csv", index=False)
    result = score_frames(pd.read_csv(processed / "truth.csv"), frame, holdout.image_id.tolist())
    pd.DataFrame(result.pop("per_image")).to_csv(output / "holdout_per_image.csv", index=False)
    write_json(output / "holdout_score.json", result)
    model_spec = {"candidate": candidate, "development_pq": selected["development_pq"], "holdout": result,
                  "selected_by": "development accumulated PQ; holdout evaluated once with frozen selection"}
    if candidate["mode"] != "classical":
        train_model(manifest, manifest.image_id.tolist(), data_root, processed, b["training"], output / "full_model")
        model_spec.update(training=b["training"], checkpoint=str(output / "full_model/model.pt"))
    write_json(output / "model.json", model_spec)
    print(json.dumps(model_spec), flush=True)
