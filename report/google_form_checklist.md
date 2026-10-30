# Final Google Form and release checklist

Prepared 30 October 2026, Asia/Singapore. This is a preparation checklist, **not** a completed form or a claim of eligibility. Participant identity, team membership, contact information and any attestations must be supplied by the participant. No form has been filled or submitted by the agent.

Official entry point: [competition Overview → Final Submission](https://www.kaggle.com/competitions/filament-segmentation-2026/overview). The exact Google Form link read earlier from that section is:

[Organizer final-submission form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform?usp=dialog)

The form's live fields have not been inspected in this preparation task. The items below distinguish requirements already read from the official Overview from participant information that still needs human completion. Do not guess additional mandatory fields.

## Required deliverables from the official Overview

- [ ] **One four-page technical-report PDF**, using the [organizer's Overleaf template](https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe). Draft source: `report/v4_technical_report_draft.md`. Compile into the requested template and check that the final PDF is exactly four pages and legible; this Markdown draft has not itself established pagination.
- [ ] **Public Git-based repository URL.** The committee must access the complete code without permission requests. Fill the participant-approved final URL: `[public repository URL]`. No GitHub repository was created by this task.
- [ ] **Pinned `requirements.txt`**, including all packages used by the released pipeline. Retain the exact Kaggle runtime freeze and the wheel/source provenance needed for offline execution. Explain installation and hardware assumptions.
- [ ] **A Jupyter notebook illustrating the entire pipeline.** Include data acquisition instructions, preprocessing, split construction, training, instance conversion, evaluation and final inference. The final notebook must point to the released selected model and use the exact code/config version, not an interrupted local run.
- [ ] **Selected model weights or parameters stored in the repository.** For ML/DL selections, provide the trained weights and load them in the notebook so predictions reproduce without retraining. For a classical selection, provide the selected parameter file and complete preprocessing/postprocessing code. Required artifacts must not depend on private or expiring download permissions.
- [ ] **Complete reproduction materials.** Include source, scripts, configurations, inference model references and any necessary packages/assets. Raw competition data are obtained through Kaggle under their own terms rather than redistributed in the public repository.
- [ ] **Google Form submission by the participant.** All participants seeking final evaluation must complete the organizer's form. File upload, repository link and personal/contact entries must match the actual released materials.

## Scientific and provenance content to include in the four-page PDF

- [ ] Describe grayscale H-alpha input, original 2048² resolution, official training polygons and independent annotators.
- [ ] Describe the temporal blocks, shared-image grouping, 27-day embargo and exact frozen development protocol.
- [ ] Preserve the v3 holdout as its one frozen evaluation. State explicitly if any later use of its labels has downgraded it to development.
- [ ] Describe the complete preprocessing, normalization, architecture, crop sampling, losses, optimizer, training budget, reconstruction and instance aggregation.
- [ ] State the official strict `IoU > 0.5`, per-annotator, global accumulated PQ convention. Distinguish it from alternative one-to-one metrics and any numeric compatibility version.
- [ ] Report all predefined V4 controls, PQ, TP/FP/FN, period performance, compute cost and failures, including unsuccessful experiments.
- [ ] Replace every V4 and platform-result placeholder with values copied from canonical artifacts. Do not invent a Public Score while the receipt is pending.
- [ ] Disclose no external MAGFiLO labels, no test-label lookup, no manually created test masks, no duplicate-instance gaming and no leaderboard-driven threshold scan.
- [ ] Attribute MIT code and separately identify dependency/model licenses. Do not assume a repository's MIT license covers Ultralytics or detector weights.
- [ ] Present public-reference scores only under their own data/protocol/scorer conditions; do not compare them directly with this project's frozen temporal holdout.
- [ ] Distinguish same-family provisional review from independent validation.

## Participant information to complete personally

These are preparation prompts, not claims about the current form's exact field schema.

- [ ] Participant and team name: `[participant to supply]`.
- [ ] Kaggle account/team URL and final selected submission IDs: `[participant to supply]`.
- [ ] Team-member names, affiliations and contact email requested by the live form: `[participant to supply]`.
- [ ] Confirm one account/team, maximum five members, applicable eligibility and no prohibited private cross-team code/data sharing.
- [ ] Confirm the participant understands and can explain the complete method, as required for AI/AutoML-assisted solutions.
- [ ] Complete any actual form attestation and contact/repository-access details personally after reviewing them. Do not supply bank/tax information through this technical-report checklist; any later prize paperwork is a separate organizer process.

## Submission and release records

- [ ] Link the P0 Kaggle submission receipt to CSV SHA-256 `c8e6ac199a3fbe429497d36d0be1973026aad093e1932ba65f306a7fbdc050f0`. Initial upload ID: **57795106**; the platform accepted it as **COMPLETE**, with **Public Score 0.15**; canonical record: `docs/official_submission_v3.md`.
- [ ] The local artifact covers all 180 image IDs in the inference manifest: 175 with 1,312 nonempty instance rows and five with naturally zero detections. Use the production receipt to establish acceptance of that CSV convention; do not add dummy masks for coverage.
- [ ] Respect at most five submissions/day and at most two selected final submissions. Record any official score as a submission check, not as a threshold/model-selection signal.
- [ ] Store final PDF hash, repository commit/tag, environment/weight hashes, selected configuration, submission ID, platform status and final Form confirmation in a new release record. Keep all v3 raw evidence unchanged.
- [ ] Keep the public repository accessible immediately after close and through the winner announcement, as required by the official open-access policy. Ensure any code-publication obligations on the competition forum/notebook are handled by the participant under the rules; this checklist does not post a message.
- [ ] Resolve the official deadline inconsistency before final entry. The previously read site header showed **15 November 2026, 14:00 Singapore time**, while Overview prose stated **15 November 2026, 11:59 pm CDT**. Use the earlier displayed deadline for planning until the organizer clarifies; do not infer a corrected date/time.

## Final handoff

Before the participant submits the form, the reviewable package should contain the four-page PDF, working public repository URL, notebook, selected weights/parameters, pinned requirements, canonical experiment table and official Kaggle receipt. Mark each unready item as pending rather than presenting this draft as a final submission.
