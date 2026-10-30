"""Single real-case regression for the newly observed radial sampling error.

No model execution, training, PQ calculation or repeated V3/V4 evaluation.
The bounded search is only development_ids[50:75]; it stops at the first empty
256-sample ring, and reuses that image for one native-profile repair witness.
"""

import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from solarseg import v4_model, v4_photometry


ROOT = Path(__file__).resolve().parents[1]


def test_observed_empty_ring_is_repaired_at_native_resolution(monkeypatch, tmp_path):
    cv2.setNumThreads(1)
    output = tmp_path / "p4_radial_failure.json"
    assert not output.exists(), "This new real-case regression has already been recorded."
    ids = json.loads((ROOT / "configs/v4_splits.json").read_text())["development_ids"][50:75]
    searched = []
    for image_id in ids:
        image = cv2.imread(str(ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images" / f"{image_id}.jpeg"), 0)
        disk, geometry = v4_model.solar_geometry(image)
        sample_geometry = {"center": [(value + .5) / 8 - .5 for value in geometry["center"]],
                           "radius": geometry["radius"] / 8}
        rho_sample = v4_model.radial_coordinate(256, 256, sample_geometry)
        ring = np.floor(rho_sample * 128).astype(np.int32)
        sample_disk = cv2.resize(disk, (256, 256), interpolation=cv2.INTER_NEAREST) > 0
        selected = sample_disk & (ring < 128)
        sample_counts = np.bincount(ring[selected], minlength=128)
        searched.append(image_id)
        empty = np.flatnonzero(sample_counts == 0)
        if len(empty):
            break
    assert len(empty) > 0, "No observed empty-ring case found in the specified development interval."

    capture = {}
    native_profile = v4_photometry._native_profile
    def record_native_profile(*arguments):
        profile, counts, rho = native_profile(*arguments)
        capture.update(profile=profile, counts=counts)
        return profile, counts, rho
    monkeypatch.setattr(v4_photometry, "_native_profile", record_native_profile)
    started = perf_counter()
    corrected = v4_photometry.flatten_limb_native(image, disk, geometry)
    seconds = perf_counter() - started
    assert corrected.shape == (2048, 2048) and corrected.dtype == np.float32
    assert np.isfinite(corrected).all()
    assert np.all(corrected[disk == 0] == 0)
    assert corrected.min() >= 0 and corrected.max() <= 1

    result = {
        "scope": "first observed development case in fixed ids[50:75]; no holdout or test labels",
        "searched_image_ids_in_order": searched,
        "image_id": image_id,
        "opencv_version": cv2.__version__,
        "geometry": geometry,
        "original_256_sample": {"shape": [256, 256], "rings": 128,
            "ring_pixel_counts": sample_counts.tolist(), "empty_ring_indices": empty.tolist(),
            "proof": "An empty selected sample makes np.median(empty) NaN; later interpolation spreads NaN and uint8 cast hides it as invalid feature pixels."},
        "native_repair": {"shape": [2048, 2048], "rings": 128,
            "ring_pixel_counts": capture["counts"].tolist(),
            "ring_profile_medians_0_1": capture["profile"].tolist(),
            "min_ring_pixel_count": int(capture["counts"].min()),
            "all_profile_medians_observed_and_finite": True,
            "corrected_all_finite": True, "corrected_min": float(corrected.min()),
            "corrected_max": float(corrected.max()), "seconds_one_case": seconds},
        "code": "src/solarseg/v4_photometry.py; frozen src/solarseg/v4_model.py unchanged",
        "validation": "one bounded real-input regression only; no CNN, training, inference, scoring or old tests",
        "effect_on_results": "Both frozen flat_radius result arms are invalid preprocessing controls. Their PQ cannot adjudicate limb correction. Raw arms do not call flatten_limb and retain their recorded status.",
        "gpu_policy": "Stop this round; native repair is a reference implementation not a validated GPU experiment.",
        "interpretation": "same-family analysis / provisional; one repaired real case does not establish all-image coverage or performance",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    empty_text = ", ".join(map(str, empty))
    report = f'''# V4 radial photometry failure: observed real case and isolated repair

Recorded 2026-10-30. Same-family analysis / provisional. Frozen V4 source,
training records, predictions and scores are preserved; no evaluation is rerun.

## Observed failure

The first failure found in the explicitly requested development_ids[50:75]
interval is **{image_id}**. It was found after reading {len(searched)} development
image(s), stopping at the first failing sample. No holdout/test labels are used.

Otsu-filled disk geometry: center {geometry['center']}, enclosing radius
{geometry['radius']:.6f} native pixels. The old estimator downsamples to 256²
and splits the roughly radius/8-pixel disk into 128 rings. Empty ring indices:
**{empty_text}**. Their selected pixel count is exactly zero, recorded in
`p4_radial_failure.json`. `median(empty)` is undefined, producing NaN; interpolating
that profile carries invalid values into the corrected image, and uint8 conversion
does not make the photometry valid. The failure precedes model training/inference.

## Isolated repair

`src/solarseg/v4_photometry.py::flatten_limb_native` estimates all 128 ring medians
from original 2048² disk pixels. It sorts ring assignments once and takes medians
within the resulting contiguous pixel groups. Every ring must be nonempty; the
function explicitly fails on empty rings, non-finite medians or non-positive
division medians. It never fills a missing ring or substitutes a default.
Interpolation occurs only between actually observed native medians.

The same image already held in memory was corrected once. Minimum native-ring
population: **{int(capture['counts'].min())} pixels**. All medians were finite and
the complete 2048² output was finite, within [0,1], and zero outside the disk.
Single-case correction cost: **{seconds:.4f} seconds** with one OpenCV thread.
There was no local CNN execution, no batch image correction, and no rerun of the
previous three synthetic contracts. The new regression is
`tests/test_v4_photometry.py`.

## Consequence for the experimental table

Both frozen **flat_radius_cc** and **flat_radius_group** arms have invalid
preprocessing. Preserve their raw PQ/counts/cost as the historical output, but mark
the comparison **invalid** rather than concluding limb correction helps or hurts.
The best reported flat-group PQ (~0.139627) is not an admissible normalization
comparison. The raw/radius arms do not call this estimator and are not invalidated
by this specific failure; their own measured limitations still apply.

This reference repair has passed one finite-output regression, not a fresh GPU
experiment or an all-development-image validation. No corrected V4 performance
claim is made. This round's GPU gate stays **stop**; do not relaunch training or
scan thresholds to compensate for the invalid arm.
'''
    (tmp_path / "p4_radial_failure.md").write_text(report)
    print(json.dumps({"image_id": image_id, "empty_256_rings": empty.tolist(),
                      "min_native_ring_count": int(capture["counts"].min()),
                      "native_seconds": seconds}))
