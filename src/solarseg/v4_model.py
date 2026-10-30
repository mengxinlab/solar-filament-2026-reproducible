"""V4 native-resolution crop experiment, independent of frozen v3 code.

Configuration keys are all required: variant (``raw_radius`` or ``flat_radius``),
seed, device, epochs, steps_per_epoch, batch_size, lr, base_channels, num_workers,
crop_size, positive_fraction, positive_jitter, inference_batch_size. This fixed
experiment uses crop_size=512, positive_fraction=0.5 and num_workers=0. The
radius channel is clip(distance / solar_radius, 0, 1.5) / 1.5. Both arms use the
same crop coordinates and D4 schedule for equal seeds and train-ID order.

Only segmentation polygons define targets and foreground sampling centers.
Original annotation bboxes, spines and categories are not read as features.
"""

from collections import defaultdict
import json
from pathlib import Path
import random
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .neural import UNet


VARIANTS = ("raw_radius", "flat_radius")


def solar_geometry(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Otsu largest bright region, filled exterior, and enclosing solar circle."""
    _, bright = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    disk = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contours, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    disk[:] = 0
    cv2.drawContours(disk, contours, -1, 1, cv2.FILLED)
    (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
    return disk, {"center": [float(cx), float(cy)], "radius": float(radius)}


def radial_coordinate(height: int, width: int, geometry: dict, x0: int = 0, y0: int = 0) -> np.ndarray:
    yy, xx = np.ogrid[y0 : y0 + height, x0 : x0 + width]
    cx, cy = geometry["center"]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32) / geometry["radius"]


def flatten_limb(image: np.ndarray, disk: np.ndarray, geometry: dict) -> np.ndarray:
    """Divide by a 128-ring median profile estimated on a 256² image."""
    sample = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    sample_geometry = {
        "center": [(value + .5) / 8 - .5 for value in geometry["center"]],
        "radius": geometry["radius"] / 8,
    }
    rho_sample = radial_coordinate(256, 256, sample_geometry)
    ring = np.floor(rho_sample * 128).astype(np.int32)
    sample_disk = cv2.resize(disk, (256, 256), interpolation=cv2.INTER_NEAREST) > 0
    profile = np.asarray([np.median(sample[(ring == index) & sample_disk]) for index in range(128)], dtype=np.float32)
    rho = radial_coordinate(*image.shape, geometry)
    background = np.interp(rho, (np.arange(128) + .5) / 128, profile)
    flattened = np.clip(image.astype(np.float32) / 255 / background, 0, 1.5) / 1.5
    flattened[disk == 0] = 0
    return flattened.astype(np.float32)


def prepare_cache(data_root: Path, manifest: pd.DataFrame, dev_ids: list[str], cache_dir: Path) -> dict:
    """Prepare native 2048² images/targets only for the explicitly provided IDs."""
    cache_dir.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "flat", "masks"):
        (cache_dir / name).mkdir()
    selected = manifest.set_index("image_id").loc[dev_ids]
    official = json.loads((data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    files = set(selected.file_name)
    records = {record["id"]: record for record in official["images"] if record["file_name"] in files}
    record_by_file = defaultdict(list)
    for record in records.values():
        record_by_file[record["file_name"]].append(record)
    instances = defaultdict(list)
    for annotation in official["annotations"]:
        if annotation["image_id"] in records:
            instances[annotation["image_id"]].append((annotation["id"], annotation["segmentation"]))
    del official
    geometries = {}
    fractions = []
    truth_rows = []
    started = perf_counter()
    for index, (image_id, row) in enumerate(selected.iterrows(), 1):
        image = cv2.imread(str(data_root / "train/train_images" / row.file_name), cv2.IMREAD_GRAYSCALE)
        disk, geometry = solar_geometry(image)
        target = np.zeros((2048, 2048), dtype=np.float32)
        centroids = []
        image_records = record_by_file[row.file_name]
        for record in image_records:
            rles = []
            for annotation_id, polygon in instances[record["id"]]:
                rle = coco_mask.merge(coco_mask.frPyObjects(polygon, 2048, 2048))
                rles.append(rle)
                truth_rows.append({"filament_id": f'{record["id"]}_{annotation_id}',
                                   "segmentation_rle": rle["counts"].decode("ascii")})
                x, y, width, height = coco_mask.toBbox(rle)
                centroids.append([float(x + width / 2), float(y + height / 2)])
            target += coco_mask.decode(coco_mask.merge(rles))
        target /= len(image_records)
        flattened = flatten_limb(image, disk, geometry)
        cv2.imwrite(str(cache_dir / "raw" / f"{image_id}.png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(cache_dir / "flat" / f"{image_id}.png"), np.rint(flattened * 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(cache_dir / "masks" / f"{image_id}.png"), np.rint(target * 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        fraction = float(target.mean())
        fractions.append(fraction)
        geometries[image_id] = dict(geometry, centroids=centroids, annotation_sets=len(image_records),
                                   positive_mask_fraction=fraction, source_filename=row.file_name)
        if index % 25 == 0:
            print(json.dumps({"cache_images": index, "total": len(selected), "seconds": perf_counter() - started}), flush=True)
    (cache_dir / "geometry.json").write_text(json.dumps(geometries, indent=2) + "\n")
    pd.DataFrame(truth_rows, columns=["filament_id", "segmentation_rle"]).to_csv(cache_dir / "dev_truth.csv", index=False)
    summary = {"images": len(selected), "image_ids": dev_ids, "resolution": [2048, 2048],
               "target": "native COCO segmentation union per annotator, equal annotator consensus",
               "foreground_sampling_centers": "centers of bboxes computed from segmentation RLE; annotation bbox field unused",
               "mean_positive_fraction": float(np.mean(fractions)), "seconds": perf_counter() - started}
    (cache_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


class CropUNet(UNet):
    """Same v3 base architecture with a two-channel first convolution."""
    def __init__(self, base_channels: int):
        super().__init__(base_channels)
        self.enc1[0] = nn.Conv2d(2, base_channels, 3, padding=1, bias=False)


class NativeCrops(Dataset):
    def __init__(self, train_ids: list[str], cache_dir: Path, config: dict):
        assert config["variant"] in VARIANTS
        assert config["crop_size"] == 512
        assert config["positive_fraction"] == .5
        assert config["num_workers"] == 0
        self.ids = train_ids
        self.config = config
        self.rng = np.random.default_rng(config["seed"])
        geometries = json.loads((cache_dir / "geometry.json").read_text())
        self.geometry = {image_id: geometries[image_id] for image_id in train_ids}
        folder = "raw" if config["variant"] == "raw_radius" else "flat"
        self.images = {image_id: cv2.imread(str(cache_dir / folder / f"{image_id}.png"), 0) for image_id in train_ids}
        self.masks = {image_id: cv2.imread(str(cache_dir / "masks" / f"{image_id}.png"), 0) for image_id in train_ids}
        self.draws = []
        self.epoch = 0

    def __len__(self):
        return self.config["steps_per_epoch"] * self.config["batch_size"]

    def __getitem__(self, index):
        image_id = self.ids[int(self.rng.integers(len(self.ids)))]
        geometry = self.geometry[image_id]
        foreground_centered = index % 2 == 0
        if foreground_centered:
            cx, cy = geometry["centroids"][int(self.rng.integers(len(geometry["centroids"])))]
            jitter = self.config["positive_jitter"]
            cx += int(self.rng.integers(-jitter, jitter + 1))
            cy += int(self.rng.integers(-jitter, jitter + 1))
        else:
            theta = float(self.rng.uniform(0, 2 * np.pi))
            radius = geometry["radius"] * np.sqrt(float(self.rng.uniform()))
            cx = geometry["center"][0] + radius * np.cos(theta)
            cy = geometry["center"][1] + radius * np.sin(theta)
        crop = self.config["crop_size"]
        x0 = int(np.clip(np.rint(cx - crop / 2), 0, 2048 - crop))
        y0 = int(np.clip(np.rint(cy - crop / 2), 0, 2048 - crop))
        intensity = self.images[image_id][y0:y0 + crop, x0:x0 + crop].astype(np.float32) / 255
        radial = np.clip(radial_coordinate(crop, crop, geometry, x0, y0), 0, 1.5) / 1.5
        image = np.stack((intensity, radial))
        target = self.masks[image_id][y0:y0 + crop, x0:x0 + crop].astype(np.float32) / 255
        rotations = int(self.rng.integers(4))
        flipped = bool(self.rng.integers(2))
        image = np.rot90(image, rotations, axes=(-2, -1))
        target = np.rot90(target, rotations)
        if flipped:
            image = np.flip(image, axis=-1)
            target = np.fliplr(target)
        self.draws.append({"epoch": self.epoch, "sample": index, "image_id": image_id,
                           "foreground_centered": foreground_centered,
                           "center_x": float(cx), "center_y": float(cy), "x0": x0, "y0": y0,
                           "rot90": rotations, "horizontal_flip": flipped,
                           "positive_fraction": float(target.mean())})
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(target[None]))


def train_variant(manifest: pd.DataFrame, train_ids: list[str], cache_dir: Path, config: dict, output_dir: Path) -> dict:
    """Fit a fixed crop variant on explicit training IDs and persist every draw."""
    manifest.set_index("image_id").loc[train_ids]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "train_ids.json").write_text(json.dumps(train_ids, indent=2) + "\n")
    dataset = NativeCrops(train_ids, cache_dir, config)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
                        generator=torch.Generator().manual_seed(config["seed"]))
    model = CropUNet(config["base_channels"]).to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    history = []
    model.train()
    total_started = perf_counter()
    for epoch in range(config["epochs"]):
        dataset.epoch = epoch + 1
        started = perf_counter()
        losses = []
        bce_losses = []
        dice_losses = []
        positive_fractions = []
        foreground_probability_sums = []
        foreground_target_masses = []
        background_probability_sums = []
        background_target_masses = []
        for image, target in loader:
            image, target = image.to(config["device"]), target.to(config["device"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            probability = logits.sigmoid()
            intersection = (probability * target).sum(dim=(1, 2, 3))
            denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
            dice = (1 - (2 * intersection + 1) / (denominator + 1)).mean()
            bce = F.binary_cross_entropy_with_logits(logits, target)
            loss = bce + dice
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            bce_losses.append(float(bce.detach()))
            dice_losses.append(float(dice.detach()))
            positive_fractions.append(float(target.mean()))
            foreground_probability_sums.append(float((probability.detach() * target).sum()))
            foreground_target_masses.append(float(target.sum()))
            background_probability_sums.append(float((probability.detach() * (1 - target)).sum()))
            background_target_masses.append(float((1 - target).sum()))
        item = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "bce": float(np.mean(bce_losses)),
                "soft_dice_loss": float(np.mean(dice_losses)), "positive_fraction": float(np.mean(positive_fractions)),
                "foreground_probability": sum(foreground_probability_sums) / sum(foreground_target_masses),
                "background_probability": sum(background_probability_sums) / sum(background_target_masses),
                "seconds": perf_counter() - started}
        history.append(item)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        pd.DataFrame(dataset.draws).to_csv(output_dir / "sample_draws.csv", index=False)
        print(json.dumps({"variant": config["variant"], "run": str(output_dir), **item}), flush=True)
    checkpoint = output_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config, "train_ids": train_ids}, checkpoint)
    return {"checkpoint": str(checkpoint), "train_images": len(train_ids), "config": config,
            "seconds": perf_counter() - total_started, "history": history}


def load_variant(checkpoint: Path, device: str) -> nn.Module:
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = CropUNet(saved["config"]["base_channels"]).to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


@torch.inference_mode()
def predict_full(model: nn.Module, image: np.ndarray, device: str, config: dict) -> np.ndarray:
    """Blend 25 native-resolution tile probabilities before instance conversion."""
    disk, geometry = solar_geometry(image)
    if config["variant"] == "raw_radius":
        intensity = image.astype(np.float32) / 255
    else:
        intensity = np.rint(flatten_limb(image, disk, geometry) * 255) / 255
    radial = np.clip(radial_coordinate(2048, 2048, geometry), 0, 1.5) / 1.5
    channels = np.stack((intensity, radial)).astype(np.float32)
    origins = [(x, y) for y in (0, 384, 768, 1152, 1536) for x in (0, 384, 768, 1152, 1536)]
    one_dimensional = np.hanning(514)[1:-1].astype(np.float32)
    weight = one_dimensional[:, None] * one_dimensional[None, :]
    numerator = np.zeros((2048, 2048), dtype=np.float32)
    denominator = np.zeros((2048, 2048), dtype=np.float32)
    batch_size = config["inference_batch_size"]
    for offset in range(0, len(origins), batch_size):
        current = origins[offset:offset + batch_size]
        batch = np.stack([channels[:, y:y + 512, x:x + 512] for x, y in current])
        probabilities = model(torch.from_numpy(batch).to(device)).sigmoid()[:, 0].cpu().numpy()
        for (x, y), probability in zip(current, probabilities):
            numerator[y:y + 512, x:x + 512] += probability * weight
            denominator[y:y + 512, x:x + 512] += weight
    result = np.clip(numerator / denominator, 0, 1)
    result[disk == 0] = -1
    return result
