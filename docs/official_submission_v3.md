# First official competition submission

- Account: `mengxinlab`; competition: `filament-segmentation-2026`.
- Submission ID: **57795106**.
- Platform state: **COMPLETE**, with no error description.
- Public Score: **0.15**, exactly as returned by Kaggle.
- Upload timestamp: 2026-10-30 14:13:47 Singapore (06:13:47 UTC).
- File: `submissions/final_v1.csv`, 760,205 bytes.
- SHA-256: `c8e6ac199a3fbe429497d36d0be1973026aad093e1932ba65f306a7fbdc050f0`.
- Raw upload receipt: `runs/v4/p0_official_submission/upload_receipt.json`.
- Raw completion response: `runs/v4/p0_official_submission/platform_detail_03.json`.
- Full one-time contract check: `runs/v4/p0_official_submission/contract_audit.json`.
- Coverage by all 180 test IDs: `runs/v4/p0_official_submission/image_coverage.csv`.

The file contains 1,312 nonempty, mutually disjoint instance masks, each a
canonical compressed ASCII COCO RLE at 2048×2048. IDs are unique. Of the 180
test observations, 175 have instance rows and five have naturally zero
detections. They are represented by absence of instance rows, with the inference
coverage retained separately. Kaggle accepted this exact representation; no
empty masks or fabricated detections were added.

This score verifies official submission acceptance only. It is not a tuning
signal, not an independent test of v3 model selection, and not a basis for
changing thresholds, V4 candidates or the GPU investment rule. The V4 plan was
written before this response was returned. Private Score is not available.
All original v3 sidecars retain their historical `not_uploaded` state; this new
record supplies the later official event without rewriting frozen evidence.
