# Historical V4 artifacts — unselected, not a new experiment

These files are verbatim artifacts of the completed original V4 Kaggle run.
The six model.pt checkpoints cover raw_radius and flat_radius, folds 0, 1, 2.
They support traceability and historical reproduction and are **not selected
release models**. Selected v3 classical parameters remain models/final_v1/model.json.

The raw/radius experiments are valid controls but did not exceed the frozen
classical development reference. Both flat/radius aggregation arms are INVALID
because the original downsampled annulus estimator produced empty-ring NaNs.
The original decision.json is preserved unchanged, including its unadjudicated
flat comparison. It is a historical program output, not a valid model-selection
instruction. Read validity_adjudication.json and ../../docs/v4_results.md before
interpreting results. Never select a flat checkpoint from the original decision.

The native photometry reference repair in ../../src/solarseg/v4_photometry.py
passed one CPU real-case regression only, has no GPU result, and did not create
these checkpoints. Frozen v4_model.py retains the historical estimator.
No additional GPU work is scheduled.

The archive includes configs, loss/target-probability histories, exact train IDs,
sampling coordinates, per-annotator-image counts, score summaries, probability
summaries, compute, hardware, source hashes, exact package freeze and execution log.
Failure cases.json files retain numerical diagnostics without source-image pixels.

Excluded: *_oof.csv RLE prediction files, failure JPG/PNG images, native cache,
raw competition images and labels. Obtain official data through your own Kaggle
permissions; this repository does not redistribute them. The 49-image temporal
holdout was not used to develop these V4 experiments and is not re-evaluated here.
