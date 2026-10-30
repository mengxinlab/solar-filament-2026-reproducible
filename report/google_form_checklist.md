# Final entry checklist

Prepared 30 October 2026, Asia/Singapore. Author details supplied by the participant: **Xin Meng (孟昕), China Medical University (中国医科大学), China, mengxinlab@gmail.com**.

## Ready materials

- [x] Four-page technical PDF: [solar_filament_technical_report.pdf](solar_filament_technical_report.pdf). All four pages were rendered and inspected; the final page update was rendered again. SHA256: `ad7567642bdc9f97e9323033a3dfebf57f2c52558115263ecba1b2cd56cf709f`.
- [x] LaTeX source, bibliography, MIT template notice and production record: `report/latex_v4/`. The official [Overleaf read link](https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe) returned a generic editor page, so exact source identity could not be checked. The inspected MIT reference uses the same ACM sigconf competition-report structure; provenance is documented.
- [x] Public repository: https://github.com/mengxinlab/solar-filament-2026-reproducible . The final evidence update and PDF are included in the release recorded by the `v4-diagnostic-20261030` tag.
- [x] Associated public CPU code notebook, COMPLETE: https://www.kaggle.com/code/mengxinlab/solar-filament-2026-reproducible-release . Training/inference remain opt-in; no GPU rerun for publication.
- [x] Pinned `requirements.txt`, `requirements-kaggle.txt`, local historical pins, complete source and offline Kaggle launcher.
- [x] Full-pipeline `notebooks/reproduce_v4.ipynb` with explicit `RUN=False`; data acquisition, preprocessing, temporal split, training, instance conversion, evaluation and final inference instructions.
- [x] Selected classical parameters: `models/final_v1/model.json`; no learned weights are needed for the selected method. Experimental V4 checkpoints and numerical histories are archived separately with valid/invalid status.
- [x] Official first submission: **57795106 / COMPLETE / Public Score 0.15**. Receipt: `docs/official_submission_v3.md`. The score verifies submission acceptance only; it is excluded from tuning.
- [x] Frozen CSV SHA256: `c8e6ac199a3fbe429497d36d0be1973026aad093e1932ba65f306a7fbdc050f0`. Its 1,312 nonempty instances cover 175 images; five zero-instance images appear in the 180-image coverage manifest and require no dummy rows.
- [x] Report includes official strict IoU>0.5, per-annotator accumulated PQ, grouped temporal validation and 27-day embargo, P2 failure diagnosis, MIT methods/licensing, complete V4 counts/periods/costs and descriptive failures.
- [x] Both flat V4 arms are **INVALID** because the sampled radial estimator produced NaNs. A separate native repair passed one CPU real-image regression only. There is no repaired GPU score.
- [x] Best valid V4 PQ 0.127734948 is below classical 0.177047225; predefined continuation gate fails, so **STOP GPU**. v3 raw evidence and its one 49-image temporal holdout evaluation remain unchanged.
- [x] Code/data/license boundaries: no redistributed raw competition images/annotations, external MAGFiLO labels, test-label lookup, public submission reuse, metric gaming or leaderboard scans. MIT notices retained; third-party dependency licenses are separate.

## Participant final actions

The official [Overview → Final Submission](https://www.kaggle.com/competitions/filament-segmentation-2026/overview) links the [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform?usp=dialog). Its read-only access attempt redirected to Google sign-in. **No form has been filled or submitted; exact live fields remain unverified.**

- [ ] Sign in and inspect the actual form fields; upload the four-page PDF and enter the public repository/notebook links as requested.
- [ ] Account: **mengxinlab**, https://www.kaggle.com/mengxinlab ; final selected submission: **57795106**. Complete any team-name/member fields personally (maximum five members under the read rules).
- [ ] Confirm eligibility, one-account/team compliance and all actual attestations. Confirm you understand and can explain the method, including AI-assisted code.
- [ ] Review final submission selection in Kaggle (at most two selected finals, at most five submissions/day). This package creates no further score probes.
- [ ] Retain the final Google Form confirmation with the PDF hash, repository tag and Kaggle receipt after submission.
- [ ] Keep repository/code publicly accessible through the winner announcement, following the official access policy.
- [ ] Resolve the observed deadline inconsistency: site header previously showed **15 November 2026, 14:00 Singapore**, while Overview prose said **15 November 2026, 11:59 pm CDT**. Plan against the earlier displayed deadline until clarified; no new interpretation is assumed.

Publishing code and preparing this checklist do not submit the final form. Routine scientific review is **same-family provisional**, not independent validation.
