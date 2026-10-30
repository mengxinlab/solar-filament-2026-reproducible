> Historical frozen v3 report, written before official upload. The later acceptance event is in [official_submission_v3.md](official_submission_v3.md): submission57795106 COMPLETE / Public0.15. Original v3 scores below are unchanged.

# Results recorded 2026-10-30

The Kaggle v3 run completed. Its downloaded canonical artifacts are under
`results/kaggle_v3/project`; this report uses existing scores, counts, and histories only.
No local training, inference, mask evaluation, or submission upload was run to
produce this report.

Selected `public_reference_sigma64` by development PQ **0.177047**.
Frozen-selection holdout PQ is **0.105604** on
49 chronological holdout images (2020–2022). The absolute drop is
**0.071443**, or **40.4%** relative to development PQ.
This is evidence of limited generalization to the later holdout in this run;
the development score should not be treated as expected hidden-test performance.

The calendar-month block bootstrap 95% percentile interval is
**[0.075674, 0.127158]**, using 20 month blocks and 2,000
resamples (seed 20261030). It groups annotators and images within each month
and is conditional on this selected model and the observed holdout months.

## All 11 development candidates

| Model | Dev PQ | 2011–13 | 2014–16 | 2017–19 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| public_reference_sigma64 | 0.177047 | 0.174206 | 0.177008 | 0.198192 | 2245 | 6084 | 5679 |
| hybrid_50 | 0.173176 | 0.174061 | 0.171342 | 0.178145 | 2231 | 6504 | 5693 |
| sigma64_precision | 0.164055 | 0.158686 | 0.168546 | 0.168899 | 2109 | 6444 | 5815 |
| multiscale32_64 | 0.059295 | 0.060233 | 0.059521 | 0.052338 | 3006 | 52556 | 4918 |
| sigma32_close2 | 0.049700 | 0.049390 | 0.051103 | 0.043552 | 3046 | 65732 | 4878 |
| sigma64_close3 | 0.032850 | 0.034716 | 0.034117 | 0.020593 | 2396 | 79931 | 5528 |
| sigma64_recall | 0.024837 | 0.025828 | 0.026015 | 0.016401 | 2316 | 104449 | 5608 |
| unet_t60_c1 | 0.018415 | 0.000000 | 0.000000 | 0.137395 | 129 | 527 | 7795 |
| unet_t30_c1 | 0.017860 | 0.001670 | 0.000000 | 0.113881 | 133 | 932 | 7791 |
| unet_t45_c3 | 0.017337 | 0.000000 | 0.000000 | 0.122689 | 122 | 609 | 7802 |
| unet_t45_c1 | 0.017315 | 0.000000 | 0.000000 | 0.122366 | 122 | 611 | 7802 |


These are locally accumulated PQ values on the [0,1] scale, including every
failed neural, postprocessing, and ensemble candidate. Development data were
used for selection; the holdout selection was frozen before its one evaluation.
The public reference candidate has prior public competition tuning. No Kaggle
leaderboard score is implied.

## Neural and hybrid findings

The best pure U-Net candidate was `unet_t60_c1` at development PQ
**0.018415**, compared with the selected candidate's
**0.177047**. It produced TP/FP/FN of
129/527/7795, versus
2245/6084/5679 for the selected candidate.
The equal-weight hybrid `hybrid_50` scored **0.173176**, with
TP/FP/FN of 2231/6504/5693.
For the best pure U-Net, fold PQ was 0.000000,
0.000000, and 0.137395; the first two folds had no
qualifying matches, and its 7795 missed annotations explain the weak
pooled instance recall. The hybrid added 420 false
positives and lost 14 true-positive pairs relative
to the selected classical candidate. Its matched-IoU sum changed by only
3.694942, so the larger false-positive
penalty outweighed that change under the official PQ formula.
Thus neither the tested pure neural configurations nor this fixed 50/50
ensemble improved the measured development instance-segmentation objective.

| Neural fold | Epochs | First training loss | Last training loss | Training seconds |
|---|---:|---:|---:|---:|
| 0 | 8 | 1.313916 | 1.021266 | 223.9 |
| 1 | 8 | 1.290475 | 1.025047 | 218.0 |
| 2 | 8 | 1.225279 | 0.757278 | 222.0 |


The histories describe optimization of training BCE plus soft Dice. They do
not contain validation loss or validation learning curves and cannot establish
overfitting. The measured PQ results support choosing the classical candidate
for this run. They do not identify whether training duration, model capacity,
probability calibration, or conversion of probability maps to instances caused
the neural and hybrid deficits; those factors were not independently isolated.
A full-training fit has no separate generalization score.

## First final submission artifact

The saved [submission CSV](../results/kaggle_v3/project/submissions/final_v1.csv) contains
**1312 rows** for **175 images with detections**.
Its recorded local status is `local_contract_pass`, format
`2048x2048 COCO compressed ASCII RLE`. The 5
images with naturally zero detections are `20170516134450Bh`, `20180507140210Th`, `20181030230030Lh`, `20200310113230Ch`, `20211214174010Mh`. No dummy instances were
created to force image coverage.

Kaggle upload status: **`not_uploaded`**. Public and private
scores are unset; there is no official score or platform-acceptance result.
CSV SHA-256: `c8e6ac199a3fbe429497d36d0be1973026aad093e1932ba65f306a7fbdc050f0`.

## Reproduction evidence

The archived [project README](../results/kaggle_v3/project/README.md) and source belong to the
completed Kaggle v3 artifact at `results/kaggle_v3/project`. Canonical evidence is in
[classical results](../results/kaggle_v3/project/runs/classical_v1/scores.json),
[neural results](../results/kaggle_v3/project/runs/unet_v1/scores.json),
[final model selection](../results/kaggle_v3/project/runs/final_v1/model.json), and
[holdout counts](../results/kaggle_v3/project/runs/final_v1/holdout_per_image.csv). Neural loss
histories and checkpoints are under `runs/unet_v1/fold0/`, `fold1/`, and
`fold2/` inside that archive. Submission metadata, runtimes, and upload status
are under its `submissions/` directory.

Leakage provenance remains in `docs/competition_contract.md`,
`docs/leakage_flag_adjudication.md`, and `data/processed/data_audit.json`.
These results do not establish performance on rare structures, fine barbs,
or a future solar cycle. A same-family provisional review does not provide
independent cross-model validation.
