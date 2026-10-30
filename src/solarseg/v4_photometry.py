"""Native-resolution photometry correction for the observed V4 empty-ring error.

The frozen V4 implementation is preserved in v4_model.py. This separate repair
estimates every one of the 128 radial medians from original 2048² disk pixels,
rather than downsampling the disk to 256² before defining the rings. All rings
must contain observations; a missing ring is an explicit contract failure, not
an instruction to interpolate missing data or substitute a default.
"""

import numpy as np


def _native_profile(image: np.ndarray, disk: np.ndarray, geometry: dict):
    """Return the 128 observed ring medians, populations and native radius map."""
    yy, xx = np.ogrid[:2048, :2048]
    cx, cy = geometry["center"]
    rho = np.hypot(xx - cx, yy - cy) / geometry["radius"]
    selected = (disk > 0) & (rho < 1)
    rings = np.floor(rho[selected] * 128).astype(np.int16)
    counts = np.bincount(rings, minlength=128)
    assert np.all(counts > 0), "Every native radial ring must contain disk pixels."
    order = np.argsort(rings, kind="stable")
    values = image[selected].astype(np.float32)[order] / 255
    ends = np.cumsum(counts)
    starts = np.concatenate(([0], ends[:-1]))
    profile = np.asarray([np.median(values[start:end]) for start, end in zip(starts, ends)], dtype=np.float64)
    assert np.isfinite(profile).all(), "Native radial medians must be finite."
    assert np.all(profile > 0), "Division requires positive observed disk medians."
    return profile, counts, rho


def flatten_limb_native(image: np.ndarray, disk: np.ndarray, geometry: dict) -> np.ndarray:
    """Correct one native 2048² image using 128 nonempty observed disk rings."""
    profile, _, rho = _native_profile(image, disk, geometry)
    # Interpolate between actual measured medians; no missing ring is filled.
    background = np.interp(rho, (np.arange(128) + .5) / 128, profile)
    corrected = np.clip(image.astype(np.float32) / 255 / background, 0, 1.5) / 1.5
    corrected[disk == 0] = 0
    return corrected.astype(np.float32)
