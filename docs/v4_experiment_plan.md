# V4 prespecified diagnostic experiments

Frozen after P1–P3 diagnosis and before V4 execution, 2026-10-30.
The Kaggle v3 artifacts, raw archive, histories, score files and local submission
sidecars remain unchanged. New work lives under `runs/v4/`, `diagnostics/v4/`
and a separate Kaggle kernel. The v3 holdout score remains the single frozen
49-image evaluation. V4 does not use its labels, fit it, or re-evaluate it.

## Evidence and hypotheses

P1 found semantic agreement with the official PQ notebook, including strict
IoU > 0.5, per-annotator comparisons and accumulated counts. Float64 versus
official float32 causes tiny rounding differences, not the neural collapse.
P2 found fold1 maximum probability <=0.154663 (all 268 images), below every
prespecified threshold; fold0 maximum <=0.503418 and therefore no detections at
0.6. Lowering the existing threshold to 0.3 still produced only six fold0 true
positive pairs. P2's 18 declared development images showed correct file/mask
alignment within uint8 target quantization. Their filament widths shrink from
roughly 8–9 native pixels to about two at whole-frame 512 input. Positive target
mass is sparse and random whole-frame draws do not address that imbalance.

The first hypothesis is that native-resolution context and deliberate foreground
exposure improve learnable filament localization. The second is that removing
the radial intensity trend improves transfer across acquisition periods. These
are testable mechanisms, not established causal conclusions from training loss.

## Exactly two training arms

Both use the existing small U-Net capacity (base16, GroupNorm/SiLU), initialized
from scratch. No external pretrained weights, YOLO models or external labels.
Both have two input channels: intensity and normalized solar radius. The sole
difference between the two new arms is raw versus radial-median-flattened
intensity. Native 512 crops are cut from original 2048 images and original
polygon-derived full-resolution consensus masks, never enlarged 512 masks.

- A: raw intensity plus radius; 50% foreground-centered jittered crops and 50%
  disk-centered random crops; exact shared D4 augmentation.
- B: the same architecture, supervision, crop coordinates, optimizer budget and
  seeds; intensity divided by a 128-ring radial median profile, with a fixed
  physical intensity clipping transformation.

Each arm/fold uses the same 384 AdamW updates (8 x 48), batch4, lr0.001 and
BCE + soft Dice as v3. This does not blindly extend epochs. V3 versus A changes
resolution, sampling and radius input together and cannot isolate each effect;
A versus B isolates the specified photometric transformation. Sample coordinates
and positive target mass are logged to make actual exposure inspectable.

## Inference and instance ablation

Use 25 overlapping native512 tiles per image, origins 0/384/768/1152/1536 in
each dimension. Blend probabilities with a strictly positive cropped Hann
window, then convert the entire 2048 map to instances once. No per-tile instance
concatenation, duplicated instances or TTA sweep.

Threshold0.45, closing radius1 and area400–100000 are fixed before V4. Compare
two instance rules on the same probability map: connected components, or
fragment grouping with fixed dilation radius6 used only to determine group
membership. Grouped output is the union of original foreground pixels; dilation
does not create predicted foreground. Primitive pieces under16 pixels are
removed in both arms before the area rule. This yields four prespecified
model/postprocessing candidates and no leaderboard parameter scan.

## Validation, budget and decision

Reuse v3's exact chronological development folds0/1/2 and 27-day training
embargo. Same physical image and all annotators stay together. Reuse the saved
split lists rather than recomputing random partitions. The 49-image fold3 remains
excluded. Record pooled PQ, TP/FP/FN/sumIoU, per-period/year counts, training and
inference seconds, peak CUDA memory, sample exposure and failure cases.

Use one ordinary Kaggle T4 session under `mengxinlab`, with a two-hour session
ceiling. Stop at the defined six training fits and four-candidate evaluation;
no automatic retries or additional parameter sweeps. Continue investing GPU only
if best pooled development PQ >=0.197047225 (v3 +0.02), at least two periods
improve versus their frozen v3 values, and every period PQ exceeds0.05. Otherwise
stop GPU spending and keep the formally submitted classical baseline. These
are compute-allocation criteria, not statistical significance claims.

Public Score from v3 submission57795106 is recorded solely as official acceptance
evidence. It must not alter any candidate, threshold, sampling rule or stopping
criterion above. Public ameyypawar results use different splits/matching and
are not compared numerically with these results.
