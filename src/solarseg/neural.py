"""Small grayscale U-Net trained from scratch on explicit training-image IDs."""

import json
from pathlib import Path
import random
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
        )


class UNet(nn.Module):
    """Three downsampling stages; base 16 gives channels 16, 32, 64, 128."""

    def __init__(self, base_channels: int):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(1, c)
        self.enc2 = DoubleConv(c, 2 * c)
        self.enc3 = DoubleConv(2 * c, 4 * c)
        self.bottleneck = DoubleConv(4 * c, 8 * c)
        self.dec3 = DoubleConv(12 * c, 4 * c)
        self.dec2 = DoubleConv(6 * c, 2 * c)
        self.dec1 = DoubleConv(3 * c, c)
        self.output = nn.Conv2d(c, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(image)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x = self.bottleneck(F.max_pool2d(x3, 2))
        x = self.dec3(
            torch.cat((F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), x3), dim=1)
        )
        x = self.dec2(
            torch.cat((F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), x2), dim=1)
        )
        x = self.dec1(
            torch.cat((F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), x1), dim=1)
        )
        return self.output(x)


class _TrainingImages(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        train_ids: list[str],
        data_root: Path,
        cache_dir: Path,
        size: int,
        samples_per_epoch: int,
    ):
        rows = manifest.set_index("image_id").loc[train_ids]
        self.files = [
            (
                data_root / "train" / "train_images" / row.file_name,
                cache_dir / "masks" / f"{image_id}.png",
            )
            for image_id, row in rows.iterrows()
        ]
        self.size = size
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.files[np.random.randint(len(self.files))]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rotations = np.random.randint(4)
        image = np.rot90(image, rotations)
        mask = np.rot90(mask, rotations)
        if np.random.randint(2):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        mask = np.ascontiguousarray(mask, dtype=np.float32) / 255.0
        return torch.from_numpy(image[None]), torch.from_numpy(mask[None])


def train_model(
    manifest: pd.DataFrame,
    train_ids: list[str],
    data_root: Path,
    cache_dir: Path,
    config: dict,
    output_dir: Path,
) -> dict:
    """Train only ``train_ids`` and save weights, configuration, IDs, and loss."""
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = config["device"]
    model = UNet(config["base_channels"]).to(device)
    dataset = _TrainingImages(
        manifest,
        train_ids,
        data_root,
        cache_dir,
        config["size"],
        config["steps_per_epoch"] * config["batch_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    history = []
    model.train()
    for epoch in range(config["epochs"]):
        started = perf_counter()
        loss_sum = 0.0
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            probability = logits.sigmoid()
            intersection = (probability * target).sum(dim=(1, 2, 3))
            denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
            dice_loss = (1 - (2 * intersection + 1) / (denominator + 1)).mean()
            loss = F.binary_cross_entropy_with_logits(logits, target) + dice_loss
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss_sum / config["steps_per_epoch"],
                "seconds": perf_counter() - started,
            }
        )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({"run": str(output_dir), **history[-1]}), flush=True)
    checkpoint = output_dir / "model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "config": config, "train_ids": train_ids},
        checkpoint,
    )
    return {
        "checkpoint": str(checkpoint),
        "config": config,
        "train_images": len(train_ids),
        "history": history,
    }


def load_model(checkpoint: Path, device: str) -> nn.Module:
    """Load a checkpoint produced by ``train_model`` for inference."""
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = UNet(saved["config"]["base_channels"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def predict_probability(
    model: nn.Module,
    image: np.ndarray,
    device: str,
    size: int,
    tta: bool = False,
) -> np.ndarray:
    """Return 2048² probabilities; optional TTA averages I, H, V, and HV flips."""
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    image_tensor = torch.from_numpy(resized.astype(np.float32)[None, None] / 255.0).to(device)
    transforms = ((), (-1,), (-2,), (-2, -1)) if tta else ((),)
    probabilities = []
    for dimensions in transforms:
        transformed = torch.flip(image_tensor, dimensions) if dimensions else image_tensor
        probability = model(transformed).sigmoid()
        probability = torch.flip(probability, dimensions) if dimensions else probability
        probabilities.append(probability)
    probability = torch.stack(probabilities).mean(dim=0)[0, 0].cpu().numpy()
    return cv2.resize(probability, (2048, 2048), interpolation=cv2.INTER_LINEAR).astype(np.float32)
