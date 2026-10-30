# V4 controlled experiment results

Public numerical/checkpoint archive: `evidence/v4/`, including `validity_adjudication.json`. The original canonical paths below document the recorded Kaggle run; the public archive omits raw images and RLE predictions.

> **Validity adjudication:** the two `flat_radius` rows are invalid normalization controls. Actual empty 256-grid annuli produced NaN medians, followed by invalid uint8 conversion during caching; inference also encountered empty profiles. Their recorded values are retained for traceability and excluded from scientific model selection. The best valid candidate is **raw_radius_group, PQ0.127735**. No period exceeds the frozen classical reference; stop further GPU spending under the predefined gate. See `runs/v4/validity_adjudication.json` and `diagnostics/v4/p4_radial_failure.md`.

Two predefined training arms, six fits, four candidates. All scores use the same development observations and official per-annotator accumulated PQ semantics. V3 evidence is unchanged; its 49-image holdout was not used or re-evaluated. Public Score0.15 from submission57795106 is acceptance-only.

| Candidate | PQ | TP | FP | FN | 2011-13 | 2014-16 | 2017-19 |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen v3 contrast | 0.177047 | 2245 | 6084 | 5679 | 0.174206 | 0.177008 | 0.198192 |
| raw_radius_cc | 0.119969 | 1519 | 6369 | 6405 | 0.144486 | 0.104235 | 0.080279 |
| raw_radius_group | 0.127735 | 1614 | 6216 | 6310 | 0.155649 | 0.108041 | 0.086687 |
| flat_radius_cc **INVALID** | 0.127519 | 2042 | 10518 | 5882 | 0.129835 | 0.132953 | 0.096548 |
| flat_radius_group **INVALID** | 0.139628 | 2261 | 10382 | 5663 | 0.141781 | 0.144623 | 0.109522 |

## Cost and investment decision

Actual GPU: Tesla T4. Complete-run elapsed: 57.1 minutes; preprocessing 5.7 minutes. GPU training and inference timings share one session; these are elapsed costs, not measured GPU utilization or billed money.

| Arm | Training min | Validation/inference min | Peak allocated GiB |
|---|---:|---:|---:|
| raw_radius | 10.8 | 13.2 | 2.08 |
| flat_radius | 10.6 | 16.5 | 2.08 |

The unfiltered raw run record names **flat_radius_group**, but this candidate is invalid. After validity adjudication, best valid candidate is **raw_radius_group**. Prespecified GPU continuation gate: pooled PQ>=0.197047225, at least two periods improved, every period PQ>0.05. Improved periods: 0. **Continue GPU: False.** Stop additional GPU spending; retain official v3 baseline. No further run is automatically launched.

## What the comparison establishes

A versus B was intended to keep crop size, two input channels, architecture, sample coordinates, seeds and 384 updates per fold fixed and change radial intensity correction. The B implementation failed its real-data numerical contract, so this run cannot establish the effect of normalization. V3 versus V4 changes several components together: resolution, foreground sampling, radius input and reconstruction. Its difference cannot identify their separate causal contributions. CC versus grouping reuses each model's exact prediction canvas; dilation only assigns instance membership and never adds foreground pixels.

Training exposure and probability diagnostics are in `v4_training_exposure.csv`, `v4_training_history.csv` and the canonical per-arm `probability_summary.csv`. Six development failure cases per candidate were selected by existing FN/FP counts and rendered with green contours for one declared annotator and red contours for predictions. They are descriptive failures, not an additional validation set.

Canonical artifact root: `results/kaggle_v4/project`. All per-annotator counts, OOF CSVs, actual crop draws, weights, compute records and failure images remain available under `runs/v4_native/`. No V4 hidden-test score or new holdout score is claimed; the reused development set and one seed per fold limit the strength of this diagnostic comparison.
