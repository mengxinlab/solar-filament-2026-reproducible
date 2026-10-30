# Competition contract — read 2026-10-30 (Asia/Singapore)

Primary pages read in a rendered Kaggle browser:

- [Overview, Evaluation, Leaderboard Ranking, Submission File, Final Submission](https://www.kaggle.com/competitions/filament-segmentation-2026/overview)
- [Data](https://www.kaggle.com/competitions/filament-segmentation-2026/data)
- [Rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules)
- Discussion, baselines, retrieval dates and downloaded source hashes:
  [research/public_baselines_and_discussions.md](research/public_baselines_and_discussions.md)
- Exact self-evaluation source: `reference/official_evaluation/self-evaluation-notebook.ipynb`.

## Objective and data

Segment individual solar filaments in grayscale 8-bit 2048×2048 H-alpha JPEGs.
The provided archive contains 707 training images, 180 test images, and one COCO
training JSON with 1,154 annotator-image records and 8,199 instance annotations.
Multiple annotators independently label the same physical image. Segmentation
polygons are the only supervision used. No spine, chirality, bounding box,
annotator ID, station or timestamp is a predictive input. Time is used for splits.
Competition data is CC BY-NC 4.0 and restricted to competition/noncommercial
academic uses. Do not redistribute raw data in the public code repository.

## Leakage and prohibited shortcuts

The host clarification in discussion 729396 limits annotation supervision to
competition training labels. Discussion 728062 reports overlaps between test
images and the public MAGFiLO release. This project does not download the external
annotation release, resolve test IDs against external labels, use third-party
submissions or manually annotate test images. The published overlap claim is
treated as a risk report; this project does not reproduce it using external labels.

All annotators of one image stay together. Four chronological blocks and a
27-day training embargo reduce short-term solar-feature overlap. Fold 3 is locked
until all model/postprocessing choices finish. `data_audit.json` records exact
byte duplicates and a declared low-resolution perceptual risk heuristic; this
does not prove the absence of all possible leakage. Test images enter only the
unlabeled provenance audit and final frozen inference.

## Metric

Current leaderboard criterion is Panoptic Quality, updated August 7, 2026 and
rescored August 12. Older Dice references remain in parts of Overview. Official
notebook semantics: compare predictions independently against each annotator-image
group; strict IoU > 0.5 defines hits; accumulate sum of hit IoUs, TP, FP and FN
across groups; PQ = sumIoU/(TP + 0.5 FP + 0.5 FN). The notebook counts every hit
pair without a one-to-one matching solver. This project produces disjoint instance
masks and does not duplicate instances or optimize this edge case. No leaderboard
probing, empty-mask gaming or test-label retrieval is used.

Local PQ is on [0,1]. It is an implementation of the published self-evaluation
notebook, not a certification of Kaggle's private scorer. The platform score is
recorded only after an actual upload and returned result; no score is invented.

Final judging combines quantitative comparison (70%) with qualitative assessment
(30%), including shape quality, fragmentation/merging, complete pipeline description,
code quality and efficiency. Leaderboard rank alone is not the final award rubric.

## Submission

One CSV with exactly `filament_id,segmentation_rle`; each row is one predicted
instance. ID = original image stem + `_` + unique instance suffix. Counts are
compressed ASCII COCO RLE from `pycocotools`, using Fortran-order uint8 masks at
2048×2048. Store counts only, without literal enclosing quote characters. Standard
CSV quoting/escaping remains CSV syntax. Instance counts may differ from ground
truth. No fake empty instance is inserted for images with no detections.

Limits: 5 submissions/day, up to 2 selected final submissions, max 5 team members,
one account and one team per participant. No private code/data sharing between
teams. Public code must respect OSI-compatible licenses and competition sharing
rules. This project generates local files; it does not automatically join, accept
agreements, publish a repository or submit a Google form.

## Hardware and reproducibility

No fixed accelerator, notebook-only execution requirement or numeric runtime/memory
cap was found in the read competition-specific Overview/Data/Rules. Efficiency is
part of judging; absence of a stated cap is not a promise of unlimited resources.
Local runs use an existing Apple M3 Max, 48 GiB shared RAM and MPS, with CPU image
processing. No paid compute is provisioned. Runtime is recorded with each model
and inference output.

Final award eligibility requires a public Git repository, pinned package versions,
a notebook demonstrating the entire pipeline, trained weights for inference
without retraining, and a four-page technical report submitted through the linked
Google form. The site header lists Nov 15, 2026 14:00 Singapore time, while the
Important Dates prose says Nov 15 11:59 pm CDT. These disagree; use the earlier
displayed deadline for planning and obtain organizer clarification before final
entry. No clarification message has been sent by this project.

## Planned comparisons, fixed before scores

Six classical candidates vary local-background scale, threshold, closing, area and
elongation. A small U-Net is trained from scratch with BCE + soft Dice using only
training-side consensus masks. Four thresholds/closing settings and one fixed
50:50 neural/contrast blend are compared on development OOF PQ. Four-flip TTA is
fixed for neural inference. The best development candidate is frozen before one
holdout evaluation. Final fit uses all official training images after that score.
Public CPU reference parameters have prior competition tuning and are identified
as such; locked holdout performance is still conditional on publicly available
knowledge and is not advertised as a pristine benchmark.
