# V4 radial photometry failure: observed real case and isolated repair

Recorded 2026-10-30. Same-family analysis / provisional. Frozen V4 source,
training records, predictions and scores are preserved; no evaluation is rerun.

## Observed failure

The first failure found in the explicitly requested development_ids[50:75]
interval is **20110918112834Ch**. It was found after reading 11 development
image(s), stopping at the first failing sample. No holdout/test labels are used.

Otsu-filled disk geometry: center [1027.04736328125, 1024.76953125], enclosing radius
940.064392 native pixels. The old estimator downsamples to 256²
and splits the roughly radius/8-pixel disk into 128 rings. Empty ring indices:
**127**. Their selected pixel count is exactly zero, recorded in
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
population: **137 pixels**. All medians were finite and
the complete 2048² output was finite, within [0,1], and zero outside the disk.
Single-case correction cost: **0.0776 seconds** with one OpenCV thread.
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
