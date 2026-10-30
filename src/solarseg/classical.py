"""Image-only brightness-deficit baseline and connected-component instances.

Input images are 2048 x 2048 grayscale uint8 arrays. ``probability`` returns
float32 local contrast, with -1 outside the inferred, eroded solar disk. Its
values are contrast scores rather than calibrated probabilities. An arithmetic
mean of maps made from the same image preserves the outside-disk sentinel and
can be passed directly to ``instances_from_probability``.

Config keys (all required): ``sigma``, ``threshold``, ``closing_radius``,
``min_area``, ``max_area``, ``min_elongation``. Area limits are in original-image
pixels; min_area is at least 3. A closing radius of 0 disables closing and a
minimum elongation of 0 or 1 disables the shape filter. The disk follows the
reference implementation: Otsu threshold, largest bright component, filled
external contour, then erosion with a 25 x 25 elliptical kernel. No fixed solar
center or radius is assumed.

Adapted from Sijan Regmi's MIT-licensed public baseline, commit
0bff2831e96b8eea4ec6b207fec3493a313997ca:
https://github.com/regmi-sijan/solar-filament-segmentation-2026

MIT License
Copyright (c) 2026 Sijan Regmi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cv2
import numpy as np


def probability(image: np.ndarray, sigma: float = 64) -> np.ndarray:
    """Compute relative local darkness, masking the limb and off-disk pixels."""
    _, bright = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    disk = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contours, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    disk[:] = 0
    cv2.drawContours(disk, contours, -1, 1, cv2.FILLED)
    disk = cv2.erode(
        disk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )

    smooth = cv2.GaussianBlur(image.astype(np.float32) / 255, (0, 0), 1.0)
    background = cv2.GaussianBlur(smooth, (0, 0), sigma)
    # The reference's intensity floor limits the relative contrast gain near
    # the limb; it is part of the image transformation, not missing-data repair.
    contrast = np.clip((background - smooth) / np.maximum(background, 0.08), 0, 1)
    contrast[disk == 0] = -1
    return contrast


def instances_from_probability(
    probability: np.ndarray, config: dict
) -> list[np.ndarray]:
    """Turn a contrast map into disjoint full-resolution instance masks."""
    disk = (probability >= 0).astype(np.uint8)
    binary = ((probability > config["threshold"]) & (disk > 0)).astype(np.uint8)
    radius = config["closing_radius"]
    if radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary &= disk

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    accepted = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < config["min_area"] or area > config["max_area"]:
            continue
        if config["min_elongation"] > 1:
            x, y, width, height, _ = stats[label]
            yy, xx = np.nonzero(labels[y : y + height, x : x + width] == label)
            eigenvalues = np.linalg.eigvalsh(np.cov(np.stack([xx, yy])))
            elongation = np.sqrt((eigenvalues[1] + 1) / (eigenvalues[0] + 1))
            if elongation < config["min_elongation"]:
                continue
        accepted.append((labels == label).astype(np.uint8))
    return accepted


def predict_instances(image: np.ndarray, config: dict) -> list[np.ndarray]:
    """Apply the configured contrast model and instance post-processing."""
    return instances_from_probability(probability(image, config["sigma"]), config)
