# Solar Filament Segmentation Challenge 2026

Reproducible grayscale H-alpha instance segmentation for the
[Kaggle competition](https://www.kaggle.com/competitions/filament-segmentation-2026).
This release separates a frozen accepted v3 submission from prespecified V4
development experiments. It redistributes no competition images or annotations.

The frozen v3 CSV received platform status **COMPLETE**, Public Score **0.15**,
submission ID **57795106**. This result verifies official submission acceptance;
it is not used to choose thresholds, architectures or postprocessing.
See [the official submission record](docs/official_submission_v3.md).

## Reproduction on ordinary Kaggle GPU

Heavy training and inference are intended for ordinary Kaggle **Tesla T4** GPU.
Do not attach ARC data or use ARC-reserved RTX6000 resources for this competition.
Use your one competition account. Accept competition rules yourself before data
access. No script accepts agreements, changes accounts or submits the final form.

1. Obtain this public repository in a writable checkout. For the notebook's
   default path, use `/kaggle/working/solar-filament-2026-reproducible`.
2. Create the offline runtime dependency folder from the public PyPI wheel:

   ```bash
   python scripts/prepare_offline_runtime.py --owner YOUR_ACCOUNT --output offline-runtime
   kaggle datasets create -p offline-runtime
   ```

   The second command creates a private dataset under your own account. The helper
   only downloads the pinned public Linux wheel and writes dataset metadata;
   it contains no authentication token and performs no upload. Original package
   license: FreeBSD/BSD-2-Clause. The package SHA256 is recorded by the helper.
3. Build a private offline kernel snapshot:

   ```bash
   python scripts/build_kaggle_v4.py --owner YOUR_ACCOUNT \
     --slug solar-filament-v4-reproduction \
     --title "Solar Filament V4 Reproduction" \
     --output-dir outputs/v4_reproduction
   python scripts/run_kaggle.py outputs/v4_reproduction
   ```

   This ordinary-GPU SaveAndRun request has a 7,200-second session ceiling,
   internet disabled, only the Solar competition source, your runtime dataset,
   and **no private v3 kernel dependency**. If your account cannot use T4, resolve
   the account's available resources explicitly instead of silently switching.
   Notebook execution does not upload a competition submission.
4. Alternatively, open [reproduce_v4.ipynb](notebooks/reproduce_v4.ipynb), set the
   explicit checkout/data paths, and leave `RUN=False` until ready. Install
   `pycocotools==2.0.10` before the offline GPU run; the wheel is mounted from your
   runtime dataset. The GPU image originally included 2.0.11, which is deliberately
   pinned to the version used in the recorded experiments rather than substituted.

The public evidence at `evidence/v3/` contains only the fixed training manifest,
classical score JSON and per-annotator-image counts. The V4 runner uses
`--v3-root evidence/v3`; another participant needs neither the original private
v3 notebook nor its original output archive. Exact V4 split IDs and two experimental
arms are in `configs/v4_splits.json` and `configs/v4_experiments.json`.

## Selected prediction without retraining

The selected released v3 method is classical, so it requires **parameters, not
learned weights**. They are stored at `models/final_v1/model.json`. In an ordinary
Kaggle GPU session with the official competition data mounted:

```bash
PYTHONPATH=src python -m solarseg.cli infer \
  --data-root /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026 \
  --model models/final_v1/model.json \
  --device cuda --output submissions/reproduced_final_v1.csv
```

Use a fresh output filename. The frozen CSV `submissions/final_v1.csv` has 1,312
nonempty disjoint instances across 175 images; five of the 180 test observations
have naturally zero detections, represented by absence of instance rows. That
convention was accepted by Kaggle. No dummy masks are inserted for coverage.
Official platform acceptance remains distinct from local format validation.

## Protocol and evidence

- Supervision uses only the competition-provided training segmentation polygons.
  No external MAGFiLO annotations, test-label lookup or public submissions are used.
- All annotators of a physical image share its split. V4 uses the fixed development
  years 2011–2019, with a 27-day training embargo around validation observations.
- The 2020–2022 holdout retains its one frozen v3 evaluation. Its labels do not
  enter V4. The v3 development PQ 0.17705 and holdout PQ 0.10560 use different
  periods; the decline limits the performance claim.
- The host's PQ definition uses strict IoU>0.5, independent annotator-image
  comparisons and accumulated counts. See the
  [same-family provisional numerical audit](review-stage/codex/v4_p1_metric_audit.md)
  for the exact compatibility details. Predictions remain disjoint; no metric
  edge case or duplicate-instance trick is used.
- [V4 plan](docs/v4_experiment_plan.md) fixes native 512 crops, equal foreground/disk
  sampling, 384 updates per fold, raw/radius versus limb-corrected/radius channels,
  and two instance aggregation controls. Report unsuccessful arms as well.
- [V3 results](docs/results.md) and [U-Net diagnosis](diagnostics/v4/p2_unet_diagnosis.md)
  are supplied. V4 result documentation and the compiled technical PDF are added
  when canonical experiments are complete; no pending value is invented.

`requirements.txt` and `requirements-kaggle.txt` pin the packages used in the completed Kaggle GPU runtime. `requirements-local-v3.txt` preserves the earlier local development environment. Every V4 kernel saves its own full package freeze, source manifest,
hardware record, sample draws, losses, prediction CSVs, PQ counts and compute cost.
Seeds and identical configs do not promise bitwise equality across devices or
library versions. Do not install the local Torch pin over an active Kaggle image;
use the corresponding GPU runtime recipe and preserve the generated freeze.

## Attribution and final entry

MIT image-processing code is adapted from Sijan Regmi's public baseline. Amey
Pawar's MIT project is a methodological reference for native crops, normalization
and instance fusion; its validation numbers are not directly compared with this
project's temporal holdout. Full notices are in `THIRD_PARTY_NOTICES.md` and
`third_party/`. Dependency and detector/model licenses are separate; this release
does not relicense them as MIT or bundle Ultralytics/YOLO weights.

The four-page report sources are in `report/latex_v4/`. Follow
`report/google_form_checklist.md` before the participant completes the official
Google Form. Publishing this repository and generating a report do not constitute
form submission. Participant identity, contact information, attestations and any
required manual final steps remain the participant's responsibility.

Associated public competition code page: https://www.kaggle.com/code/mengxinlab/solar-filament-2026-reproducible-release . It is public and completed on CPU, with training/inference opt-in, and does not consume GPU quota or repeat frozen results.
