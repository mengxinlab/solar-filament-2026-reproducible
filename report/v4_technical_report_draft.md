# Solar Filament Instance Segmentation with Temporal Validation and Resolution-Aware Diagnostics

**Technical-report draft for Solar Filament Segmentation Challenge 2026**  
**Author:** Xin Meng (孟昕)  
**Affiliation:** China Medical University (中国医科大学), China  
**Contact:** mengxinlab@gmail.com  
**Team name:** [participant to complete if requested]  
**Evidence cutoff:** 30 October 2026, Asia/Singapore  
**Status:** v3 results are frozen; V4 experiment results must be populated from their canonical artifacts before submission. Public repository: https://github.com/mengxinlab/solar-filament-2026-reproducible . Associated public Kaggle notebook: https://www.kaggle.com/code/mengxinlab/solar-filament-2026-reproducible-release . This draft is intended for the organizer's four-page template; final pagination is a separate production step.

## Abstract

We study image-only instance segmentation of solar filaments in 2048×2048 grayscale GONG H-alpha observations. Our reproducible pipeline uses only the competition's training polygons, separates observations by time, keeps independent annotators together, and evaluates instances using the host's published Panoptic Quality (PQ) semantics. In a completed Kaggle run, a local-contrast baseline achieved development PQ 0.177047, outperforming a small U-Net and a fixed neural–contrast mixture. The selected baseline obtained PQ 0.105604 in one frozen assessment on 49 later observations, indicating limited temporal generalization. Diagnostics identify low neural responses, sparse positive pixels, and loss of filament scale after whole-frame downsampling; they do not establish overfitting or input–target misalignment. We therefore define a small V4 study of native-resolution training and deterministic instance aggregation. All V4 results remain [populate from canonical V4 artifact]. The official submission receipt is recorded separately from local validation and is not used to tune parameters.

## 1. Problem, data and evidence boundaries

Solar filaments are dark, elongated and often finely structured regions in H-alpha imagery. The challenge requires one mask per physical filament, rather than a single foreground map. Fine barbs, weak contrast, disconnected visible fragments and nearby dark structures make semantic accuracy and instance identity distinct problems. Our objective is a reproducible inference procedure with honest validation, rather than optimization of leaderboard edge cases.

The official data contain 707 training observations and 180 test observations, each an 8-bit, 2048×2048 grayscale JPEG. The COCO-style training file contains 1,154 annotator–image records and 8,199 instance annotations. A physical observation can have several independent annotations. We use segmentation polygons only. Filament spines, bounding boxes, chirality classes, annotator IDs and timestamps are not predictive inputs; timestamps determine partitions.

External MAGFiLO annotations are excluded. The host restricts annotation supervision to the competition training labels, and public discussion reports overlap between the released MAGFiLO data and competition test imagery. We do not match test IDs to external annotations, obtain test labels, manually annotate the test set, use public submissions, or exploit scoring pathologies. An exact-byte audit found no duplicates in the supplied images. A low-resolution similarity heuristic flagged a small number of pairs; inspection of the flagged cross-fold training pairs found different high-frequency content, supporting a solar-disk-shape explanation rather than an identical-image conclusion. This audit is not proof that every possible dependency is absent.

### 1.1 Temporal protocol and the frozen holdout

We partition source images into 2011–2013, 2014–2016, 2017–2019 and 2020–2022 blocks, containing 279, 268, 111 and 49 images. All annotators of one source image stay in the same block. The first three blocks form the development folds. A 27-day embargo removes training observations close to any validation observation; development observations close to the fourth block are also excluded. Exact image IDs and the resulting training/validation membership are archived.

The v3 configuration was selected using development predictions, then frozen before its single assessment on the 49-image fourth block. Those labels are not used to design V4. The historical holdout score remains a frozen result; it must not be repeatedly reselected or rewritten. If later work uses its labels or failure masks for development, the subset must be explicitly relabeled as development, and later results must not be presented as an untouched holdout evaluation. Public reference parameters already reflect their author's competition-data tuning, which limits claims of pristine validation even under our temporal split.

## 2. Methods and official evaluation

### 2.1 v3 reference and neural baseline

The selected classical method adapts the MIT-licensed public implementation by Sijan Regmi. A bright solar disk is found with Otsu thresholding, the largest connected region, contour filling and erosion. A smoothed grayscale image is compared with a Gaussian local background to form relative darkness. The selected background scale is 64 original pixels, the contrast threshold is 0.04, morphological closing is disabled, retained component areas are 800–100,000 original pixels, and minimum elongation is 2. Components produce disjoint instance masks. These settings are identified as a public reference, not a novel method.

The v3 U-Net is trained from scratch with a grayscale input, GroupNorm, SiLU activations and feature widths 16/32/64/128. It receives an entire observation resized to 512×512. For each annotator, training polygons are unioned into foreground; the independent annotator unions are averaged into a soft target and resized with area interpolation. This target is used for training only. Evaluation retains each original 2048² instance separately.

Training uses BCE plus soft Dice, AdamW, fixed seeds, and corresponding dihedral transformations of images and targets. Each fold receives 384 optimizer updates: eight epochs, 48 updates per epoch, batch size four. This is 1,536 sampled whole-frame draws; it is not eight exhaustive passes through every image. There is no foreground-directed sampling or solar-disk intensity normalization in this neural baseline. Inference averages identity, horizontal, vertical and combined flips, restores probabilities to 2048², masks off-disk regions, and converts thresholded connected regions into instances. Four predefined threshold/closing combinations and a fixed 50:50 neural–contrast mixture are compared on development data.

### 2.2 PQ and instance accounting

We follow the organizer's archived self-evaluation notebook. For each annotator–image record, every ground-truth/prediction pair with **strict** IoU > 0.5 is a hit. The official code counts all hit pairs; it does not use Hungarian or greedy one-to-one assignment. False positives are zero-hit prediction columns and false negatives are zero-hit ground-truth rows. Across all annotator–image records,

\[
\mathrm{PQ}=\frac{\sum_{(g,p):\mathrm{IoU}(g,p)>0.5}\mathrm{IoU}(g,p)}{\mathrm{TP}+0.5\mathrm{FP}+0.5\mathrm{FN}}.
\]

The numerator and counts are accumulated before taking the ratio. Averaging per-image PQ or combining annotators into one GT instance set would change the official protocol. Our evaluator uses compressed COCO RLE and obtains IoUs without expanding all pairs into full-resolution tensors. A same-family provisional code audit found semantic agreement with the official notebook. It identified a minor numeric difference: the notebook uses float32 IoUs while the COCO path retains float64. A new 2/3-IoU witness differed by approximately 1.99×10⁻⁸, with identical hit counts. Frozen v3 scores are retained as recorded. Any new evaluator that reproduces notebook rounding must have a distinct recorded version.

All submitted masks are image-derived, nonempty binary masks. We do not duplicate or nest instances to multiply hit pairs. Natural zero-instance images produce no fabricated mask rows and are retained in a separate inference-coverage record. The official notebook's zero-prediction behavior is distinct from Kaggle's production CSV acceptance, which is tested by the actual formal submission.

## 3. Frozen v3 results and failure diagnosis

The completed Kaggle v3 run used a Tesla T4 with 14.6 GiB reported VRAM. No v3 artifact has been overwritten. Table 1 summarizes selected development candidates; the archive includes all eleven candidates, their period-specific results, masks, configurations and counts.

| Development candidate | PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| Public contrast reference | 0.177047 | 2,245 | 6,084 | 5,679 |
| Fixed 50:50 hybrid | 0.173176 | 2,231 | 6,504 | 5,693 |
| Best pure U-Net, threshold 0.60 | 0.018415 | 129 | 527 | 7,795 |

The reference's period PQ values are 0.174206, 0.177008 and 0.198192. The selected reference's frozen later-block assessment gives PQ **0.105604**, TP/FP/FN **49/261/226**. The absolute development-to-holdout drop is 0.071443, or 40.4%. A previously computed calendar-month block-bootstrap interval is [0.075674, 0.127158], conditional on this selected model and the observed holdout months; it is not an uncertainty interval for the hidden test score. The reference remains weak on this later distribution.

The first submission contains 1,312 predicted instances on 175 of the 180 test observations; five observations naturally produce zero instances. The frozen CSV SHA-256 is `c8e6ac199a3fbe429497d36d0be1973026aad093e1932ba65f306a7fbdc050f0`. The formal platform upload has identifier **57795106**. **Platform state: COMPLETE; Public Score: 0.15**, copied from the official response for submission 57795106. The Public Score serves only to establish formal submission behavior and is excluded from threshold or model selection.

### 3.1 What the neural diagnosis establishes

The best pure U-Net has period PQ 0, 0 and 0.137395. The diagnostic probability records explain why thresholded masks are absent: all 268 fold-1 validation maps have maximum probability at most 0.154663, below the lowest tested threshold of 0.30. All fold-0 maxima are at most 0.503418, so threshold 0.60 necessarily produces no instances. Lowering the fold-0 threshold to 0.30 still yields only 6 TP versus 201 FP and 3,725 FN. This is evidence of a low-response model, not evidence that a more permissive threshold solves the problem.

Checks on 18 prespecified development images reconstruct cached targets within 0.001960814, the expected half-step uint8 quantization error, and confirm corresponding image/target transformations. They provide no evidence of an input–target offset. In the diagnostic samples, mean soft foreground fractions for folds 0/1/2 are approximately 0.003108/0.003369/0.000872. Median native filament widths of 9.297/8.549/7.928 pixels become approximately 2.32/2.14/1.98 pixels under fourfold downsampling. Seventeen of 155 sampled instances have hard 512² areas below 25 pixels, the resized equivalent of the 400-original-pixel component cutoff. Both spatial loss and component removal can therefore matter.

Together, sparse supervision, whole-frame downsampling, short fixed update budgets and absence of disk normalization motivate controlled interventions. These observations do not isolate each factor's causal contribution. Decreasing training loss is insufficient to claim overfitting, and more epochs are not an evidence-based first remedy. Canonical P2 diagnostic evidence: **diagnostics/v4/p2_unet_diagnosis.md and diagnostics/v4/p2_unet_diagnosis.json**.

## 4. V4 study, reproducibility and limitations

### 4.1 Predefined resolution-aware comparison

We inspected the MIT repository `ameyypawar/filament-seg` at commit `81a739bb7786a44b989f5d76afbb6f9d22b5d1dd`. It uses native 512² crops, eight per image/epoch, with 70% instance-directed and 30% solar-disk sampling. Its radial limb-darkening correction estimates a smoothed median intensity profile and provides corrected intensity plus normalized radius to an ImageNet-initialized ResNet34 U-Net. Overlapping tiles are reconstructed with a Hann weighting window. Its detector fusion assigns previously unclaimed semantic foreground to confidence-filtered, dilated YOLO11-seg instance regions and drops unclaimed foreground. These are method references, not results of our V4 study.

That repository's 563/144 split groups dates randomly, without our 27-day embargo; its validation cross-fitting of postprocessing does not make the whole research process an untouched temporal assessment. Its greedy one-to-one matcher and treatment of prediction-only images also differ from the host notebook. We therefore neither inherit its evaluator as official PQ nor compare its reported validation score directly with our holdout. The full source and protocol record is `docs/research/v4_filament_seg/method_protocol_report.md`.

V4 is limited to a small, written-before-training plan. Shared conditions are the original development-image membership and 27-day embargo, native-resolution crop supervision, an explicit foreground/background sampling policy, fixed seeds/update budgets, fixed full-image reconstruction and transparent instance conversion. The contrast reference is the control. Normalization and instance aggregation are varied only through the frozen contrast matrix. Overlapping crop outputs are consolidated into one probability canvas before instance extraction; duplicated crop instances must not count as separate predictions. YOLO11-seg fusion is a method reference, not a permission to import detector code or weights without satisfying its separate license.

| Frozen V4 experiment | Development PQ | TP / FP / FN | Period PQ | GPU time / peak memory | Failure cases |
|---|---:|---|---|---|---|
| [ID and configuration from frozen V4 plan] | [populate from canonical V4 artifact] | [populate] | [populate] | [populate] | [populate] |
| [ID and configuration from frozen V4 plan] | [populate from canonical V4 artifact] | [populate] | [populate] | [populate] | [populate] |

All candidates, including failures, must be reported. A decision to spend further GPU time requires development evidence of improved instance matching or a resolved failure mode, together with acceptable compute cost. A leaderboard increase alone is not that evidence. No additional tuning decision may use the frozen holdout or Public Leaderboard.

### 4.2 Reproduction, cost and licensing

Source, configuration hashes, image memberships, loss histories, checkpoints, PQ counts and submission receipts are retained. The v3 remote environment records Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, NumPy 2.1.3, pandas 2.3.3 and pycocotools 2.0.10, plus a full package freeze. The three neural training phases took approximately 223.9, 218.0 and 222.0 seconds; these exclude preparation, validation inference, classical search and packaging. Compute totals for V4 must use the complete remote run records. No paid resource is provisioned. Ordinary Kaggle GPU is used; ARC-reserved RTX6000 hardware is excluded from this project.

Release materials comprise a public Git repository, pinned requirements, a complete pipeline notebook, selected model parameters/weights sufficient for inference without retraining, and this four-page report. The raw competition dataset is not redistributed. The classical reference and studied repository require their MIT notices and attribution. Dependencies retain their own licenses; an MIT parent repository does not relicense Ultralytics or detector weights. Any pretrained encoder or detector must have recorded provenance, accessibility and license compatibility. Code review here is same-family/provisional, not independent validation.

Current limitations are low absolute PQ, substantial false-positive/false-negative counts, thin-structure resolution loss, limited diagnostic sampling and temporal shift. The study does not establish robustness to a future solar cycle or reliable fine-barb delineation. Four-page formatting, V4 results, selected-weight availability and final Google Form fields remain release gates. Author name, affiliation and email were supplied directly by the participant for public report attribution.

## References

1. Ahmadzadeh et al. *Solar Filament Segmentation Challenge 2026*. Kaggle, 2026. https://www.kaggle.com/competitions/filament-segmentation-2026
2. Ahmadzadeh. *Self Evaluation Notebook*. https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
3. MAGFiLO dataset description. *Scientific Data*, 2024. https://doi.org/10.1038/s41597-024-03876-y
4. Kirillov et al. *Panoptic Segmentation*. CVPR, 2019. https://doi.org/10.1109/CVPR.2019.00963
5. Ronneberger et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. 2015. https://arxiv.org/abs/1505.04597
6. He et al. *Deep Residual Learning for Image Recognition*. CVPR, 2016. https://arxiv.org/abs/1512.03385
7. Regmi. *Solar Filament Segmentation 2026*, MIT code reference, commit `0bff2831e96b8eea4ec6b207fec3493a313997ca`. https://github.com/regmi-sijan/solar-filament-segmentation-2026
8. Pawar. *filament-seg*, MIT repository, inspected commit `81a739bb7786a44b989f5d76afbb6f9d22b5d1dd` (8 October 2026). License, source hashes and dependency-license record: `docs/research/v4_filament_seg/`. https://github.com/ameyypawar/filament-seg
