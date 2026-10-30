# P1 official PQ audit — 2026-10-30

- Reviewer: Codex fresh subagent `/root/p1_metric_audit`.
- Family/type: **same-family review; provisional**. This is not independent or cross-model validation.
- Scope: static correspondence to the saved host self-evaluation notebook, original polygon-to-instance conversion, grouping, aggregation, and one new numeric-precision witness.
- Primary official source: [host notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook), archived at `docs/reference/official_evaluation/self-evaluation-notebook.ipynb`, code cells 3–4.
- Project evidence: `src/solarseg/metrics.py`, `src/solarseg/rle.py`, `src/solarseg/prepare.py`, `tests/test_contracts.py`, `runs/contract_checks.txt`, and `runs/contract_check_correction.txt`.
- No v3 source, frozen results, weights, predictions or holdout labels were modified. No empirical result was rescored. No training or bulk inference was performed.

## Assessment

The local implementation matches the host notebook's **scoring semantics**. No critical or major scoring defect was found that explains the two neural folds' zero true positives. There is one minor numeric difference: the notebook rounds each IoU to float32, while the project's COCO IoU path retains float64. This should be disclosed when claiming parity; the implementations are not universally bit-identical.

## Contract correspondence

| Item | Host notebook | Project | Verdict |
|---|---|---|---|
| Instance overlap | Pairwise intersection / binary union; rows GT, columns predictions | `mask_util.iou(pred_rles, gt_rles, [0] * n_gt).T` | Consistent for non-crowd binary masks |
| Threshold | `iou_matrix > 0.5`, strict | `hits = iou > 0.5`, `metrics.py:66` | Consistent |
| Matching | Every qualifying pair is a TP; no Hungarian or greedy assignment | `hits.sum()`, `metrics.py:67` | Consistent |
| FP / FN | Zero-hit columns / zero-hit rows | Column / row sums, `metrics.py:68–69` | Consistent |
| Annotators | GT ID prefix before `_` is `<annotator>-<image>`; reuse one image's prediction set separately for each annotator | Same prefix split, then `split('-', 1)` and lookup by image, `metrics.py:34–58` | Consistent |
| Aggregation | Sum all matched IoUs and TP/FP/FN across annotator-image records; calculate one ratio | Accumulated totals, `metrics.py:82–92` | Consistent; not the mean of image PQs |
| Zero predictions | Count all GT instances as FN; no dummy prediction needed for notebook scoring | Empty `(n_gt, 0)` matrix gives FN count, `metrics.py:61–69` | Consistent |
| Prediction-only image | `get_overlap_df` enumerates only GT prefixes; predictions on images absent from GT rows are not evaluated | Loop enumerates only GT groups, `metrics.py:54` | Same limitation; do not exploit |
| No denominator | Return 0.0 | `_pq`, `metrics.py:17–19` | Consistent |
| RLE | Compressed COCO counts with fixed 2048² size | ASCII counts; Fortran uint8 encoding, `rle.py:7–20` | Consistent |
| Polygon truth | Notebook consumes already encoded instance rows | Each original annotation polygon converted with `frPyObjects` + `merge` at the image's 2048² size; one truth row per annotation and original annotator-image, `prepare.py:133–140` | No semantic union or downsampling in evaluation truth |

The 512² soft consensus union at `prepare.py:142–149` is a **training target only**. It does not replace original instance truth in PQ. Altering this target for V4 belongs to the training/instance-aggregation investigation, not a correction of the evaluator.

The local scorer's `image_ids` option filters both frames to the selected image set before the same group evaluation. The caller must pass a complete predefined validation set, rather than selecting images according to prediction success.

## Existing verification reused

The original contract run passed seven cases and had one incorrect hand-written expected score. The saved failure shows **both** project and official implementation returned `3/7`; the test had expected `1/2`. The corrected single case passed in `runs/contract_check_correction.txt` and explicitly recorded exact pairwise-IoU equality for that fixture. The test suite was **not rerun** in this audit.

Those saved cases cover asymmetric RLE orientation, perfect/disjoint masks, exactly 0.5 IoU (not a hit), repeated annotators, zero predictions, and literal execution of official code cells 3–4. The earlier arithmetic-test failure is not evidence of a model/scorer failure.

## One new precision finding

The earlier fixture used exactly representable values such as 0.75 and 0.5. One new synthetic fixture uses a 3-pixel GT and its 2-pixel subset, giving IoU `2/3`:

| Quantity | Value |
|---|---:|
| Host notebook PQ (float32 IoU, then Python sum) | 0.6666666865348816 |
| Current project PQ (COCO float64 IoU) | 0.6666666666666666 |
| Absolute difference | 1.9868214962137642e-08 |
| Both methods' TP / FP / FN | 1 / 0 / 0 |

Machine-readable evidence: `review-stage/codex/v4_p1_precision_case.json`. This was the only new execution and did not use competition labels.

For binary 2048² masks, the union is at most 4,194,304 pixels. The smallest possible nonzero distance of an integer-ratio IoU from 0.5 is at least `1/(2 * 4,194,304)`, about `1.19e-7`, exceeding float32 rounding around 0.5. Thus this representation difference does not turn an above/below-threshold integer-area IoU into the opposite threshold decision under these mask-size assumptions. It can change displayed PQ in the last decimals; it does not explain whole folds with zero hits.

Do not overwrite or rescore frozen v3 artifacts for this minor representation difference. If exact notebook numeric compatibility is required in a **new V4 evaluator**, round each COCO IoU to float32 and convert it back to float64 before summing matched values. Casting only the final aggregate is not the same operation. Record the evaluator version and retain current analytic-precision values as their own frozen historical result.

## Adjudication and required scope

| Proposed issue/action | Decision | Reason / scope |
|---|---|---|
| “Replace current matching with Hungarian assignment to match the official metric” | **REJECT** | It would diverge from the archived official code. A one-to-one diagnostic may be named separately, never substituted for official PQ. |
| “Average per-image/per-annotator PQ instead of accumulating counts” | **REJECT** | Official score is the accumulated ratio. Mean image PQ would change weights. |
| “Combine annotators into one GT instance set before official PQ” | **REJECT** | Official evaluates each annotator-image independently. |
| “Claim bit-exact official parity for all masks” | **ACCEPT correction** | Narrow claim to semantic parity; disclose float32/float64 difference. Minor severity. A new evaluator may explicitly reproduce notebook rounding. |
| “Rewrite v3 scores or rerun the holdout because of the precision finding” | **REJECT** | No selection-relevant or threshold defect found; preserve frozen evidence. |
| “Treat production acceptance of zero-instance images as proved by this notebook” | **REJECT** | Notebook scoring permits zero predictions. Kaggle's production CSV checks require the actual P0 submission receipt; that is a separate contract. |

## V4 instance and validation rules

1. Generate each instance from image-only model output. No duplicate/nested copies intended to multiply TP hits. No GT-informed fragmentation, merging or instance count.
2. Consolidate overlapping crop predictions deterministically into a single probability canvas before connected-component extraction, or use a predefined class-agnostic instance-fusion rule. Log overlap/deduplication behavior. Do not preserve duplicate crop instances as separate identical submissions.
3. Threshold, closing, minimum area and fusion parameters must be prespecified or selected using development folds only. Keep every image/annotator associated with one physical observation in one dated validation group, with the existing 27-day embargo.
4. Natural zero-instance output is zero rows for that image, with its image ID retained in a separate inference-coverage manifest. No artificial empty or positive dummy masks. P0's actual platform response determines any additional accepted submission convention.
5. Record TP, FP, FN and sumIoU, then report accumulated PQ for the predefined full validation set and each period. Also report zero-detection counts and fragmentation/merging diagnostics; never drop failed images.
6. The v3 49-image temporal holdout remains a single frozen assessment. Do not inspect its per-image masks/labels to design V4. If those labels are later used, explicitly relabel the subset as development and cease presenting later scores on it as untouched holdout evidence.

Missing evidence outside P1: actual production submission acceptance/Public Score (P0), neural probability/scale/alignment and instance-conversion diagnosis (P2), and any V4 experiment results. None is inferred from evaluator correspondence.
