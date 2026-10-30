"""Compressed ASCII COCO RLE for the competition's 2048 × 2048 masks."""

import numpy as np
from pycocotools import mask as mask_util


def as_rle(counts: str) -> dict:
    """Build a pycocotools RLE from competition CSV counts."""
    return {"size": [2048, 2048], "counts": counts.encode("ascii")}


def encode(mask: np.ndarray) -> str:
    """Encode a binary 2048 × 2048 mask in COCO column-major order."""
    encoded = mask_util.encode(np.asfortranarray(mask, dtype=np.uint8))
    return encoded["counts"].decode("ascii")


def decode(counts: str) -> np.ndarray:
    """Decode competition CSV counts into a uint8 2048 × 2048 mask."""
    return mask_util.decode(as_rle(counts))
