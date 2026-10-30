"""Summarize persisted evidence without rerunning training, inference, or scoring."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def generate_report(artifact_root: Path, output_dir: Path, submission: dict) -> dict:
    """Read each canonical result once and derive one month-block bootstrap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for family, relative in [("classical", "runs/classical_v1"), ("neural", "runs/unet_v1")]:
        scores = json.loads((artifact_root / relative / "scores.json").read_text())
        for name, result in scores.items():
            rows.append({"family": family, "model": name, "pq": result["pq"],
                         "fold0": result["folds"]["0"], "fold1": result["folds"]["1"],
                         "fold2": result["folds"]["2"], "tp": result["tp"],
                         "fp": result["fp"], "fn": result["fn"], "sum_iou": result["sum_iou"]})
    ordered = sorted(rows, key=lambda row: row["pq"], reverse=True)
    pd.DataFrame(ordered).to_csv(output_dir / "experiment_results.csv", index=False)
    final = json.loads((artifact_root / "runs/final_v1/model.json").read_text())
    per_image = pd.read_csv(artifact_root / "runs/final_v1/holdout_per_image.csv", dtype={"image_id": str})
    # Group all annotators and observations from the same month together.
    # This derives uncertainty from saved counts; it never decodes masks or rescoring predictions.
    per_image["month"] = per_image.image_id.str[:6]
    blocks = per_image.groupby("month")[["tp", "fp", "fn", "sum_iou"]].sum().to_numpy()
    rng = np.random.default_rng(20261030)
    samples = blocks[rng.integers(len(blocks), size=(2000, len(blocks)))].sum(axis=1)
    denominator = samples[:, 0] + .5 * samples[:, 1] + .5 * samples[:, 2]
    boot = np.divide(samples[:, 3], denominator, out=np.zeros(len(samples)), where=denominator > 0)
    ci = np.quantile(boot, [.025, .975]).tolist()
    histories = {}
    history_rows = []
    for fold in (0, 1, 2):
        fold_dir = artifact_root / "runs/unet_v1" / f"fold{fold}"
        history = json.loads((fold_dir / "history.json").read_text())
        config = json.loads((fold_dir / "config.json").read_text())
        histories[str(fold)] = {"config": config, "history": history}
        history_rows.append({"fold": fold, "epochs": len(history), "first_loss": history[0]["loss"],
                             "last_loss": history[-1]["loss"], "seconds": sum(h["seconds"] for h in history)})
    selected = next(row for row in rows if row["model"] == final["candidate"]["id"])
    pure = max((row for row in rows if row["model"].startswith("unet_")), key=lambda row: row["pq"])
    hybrid = next(row for row in rows if row["model"] == "hybrid_50")
    gap = final["development_pq"] - final["holdout"]["pq"]
    relative_gap = gap / final["development_pq"]
    holdout_images = int(per_image.image_id.nunique())
    validation = submission["validation"]
    evidence = {
        "artifact_root": str(artifact_root),
        "selected": final["candidate"], "development_pq": final["development_pq"],
        "candidate_scores": ordered, "neural_training": histories,
        "locked_holdout": final["holdout"], "holdout_images": holdout_images,
        "development_to_holdout_absolute_drop": gap,
        "development_to_holdout_relative_drop": relative_gap,
        "holdout_month_blocks": len(blocks), "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261030, "holdout_month_bootstrap_95_percentile": ci,
        "submission": submission,
        "limitations": [
            "Chronological holdout covers 2020–2022; it is not the hidden Kaggle test set",
            "Development PQ was used to select the candidate and has selection optimism",
            "Development-to-holdout degradation shows limited temporal generalization in this run",
            "Public reference parameters were previously tuned by their author on competition training data",
            "Training loss alone cannot diagnose overfitting; validation loss and learning curves were not measured",
            "The experiments do not isolate architecture, training budget, probability calibration or postprocessing as causes",
            "Bootstrap interval is conditional on the selected model and observed months; it is not Kaggle score uncertainty",
            "A same-family provisional review is not independent validation",
        ],
    }
    (output_dir / "final_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    table = "| Model | Dev PQ | 2011–13 | 2014–16 | 2017–19 | TP | FP | FN |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    table += "".join(f'| {r["model"]} | {r["pq"]:.6f} | {r["fold0"]:.6f} | {r["fold1"]:.6f} | {r["fold2"]:.6f} | {r["tp"]} | {r["fp"]} | {r["fn"]} |\n' for r in ordered)
    history_table = "| Neural fold | Epochs | First training loss | Last training loss | Training seconds |\n|---|---:|---:|---:|---:|\n"
    history_table += "".join(f'| {r["fold"]} | {r["epochs"]} | {r["first_loss"]:.6f} | {r["last_loss"]:.6f} | {r["seconds"]:.1f} |\n' for r in history_rows)

    def link(relative: str) -> str:
        return Path(os.path.relpath(artifact_root / relative, output_dir)).as_posix()

    zero_ids = ", ".join(f"`{image_id}`" for image_id in validation["images_without_detections"])
    text = f'''# Results recorded 2026-10-30

The Kaggle v3 run completed. Its downloaded canonical artifacts are under
`{artifact_root}`; this report uses existing scores, counts, and histories only.
No local training, inference, mask evaluation, or submission upload was run to
produce this report.

Selected `{final['candidate']['id']}` by development PQ **{final['development_pq']:.6f}**.
Frozen-selection holdout PQ is **{final['holdout']['pq']:.6f}** on
{holdout_images} chronological holdout images (2020–2022). The absolute drop is
**{gap:.6f}**, or **{100 * relative_gap:.1f}%** relative to development PQ.
This is evidence of limited generalization to the later holdout in this run;
the development score should not be treated as expected hidden-test performance.

The calendar-month block bootstrap 95% percentile interval is
**[{ci[0]:.6f}, {ci[1]:.6f}]**, using {len(blocks)} month blocks and 2,000
resamples (seed 20261030). It groups annotators and images within each month
and is conditional on this selected model and the observed holdout months.

## All {len(rows)} development candidates

{table}

These are locally accumulated PQ values on the [0,1] scale, including every
failed neural, postprocessing, and ensemble candidate. Development data were
used for selection; the holdout selection was frozen before its one evaluation.
The public reference candidate has prior public competition tuning. No Kaggle
leaderboard score is implied.

## Neural and hybrid findings

The best pure U-Net candidate was `{pure['model']}` at development PQ
**{pure['pq']:.6f}**, compared with the selected candidate's
**{selected['pq']:.6f}**. It produced TP/FP/FN of
{pure['tp']}/{pure['fp']}/{pure['fn']}, versus
{selected['tp']}/{selected['fp']}/{selected['fn']} for the selected candidate.
The equal-weight hybrid `hybrid_50` scored **{hybrid['pq']:.6f}**, with
TP/FP/FN of {hybrid['tp']}/{hybrid['fp']}/{hybrid['fn']}.
For the best pure U-Net, fold PQ was {pure['fold0']:.6f},
{pure['fold1']:.6f}, and {pure['fold2']:.6f}; the first two folds had no
qualifying matches, and its {pure['fn']} missed annotations explain the weak
pooled instance recall. The hybrid added {hybrid['fp'] - selected['fp']} false
positives and lost {selected['tp'] - hybrid['tp']} true-positive pairs relative
to the selected classical candidate. Its matched-IoU sum changed by only
{hybrid['sum_iou'] - selected['sum_iou']:.6f}, so the larger false-positive
penalty outweighed that change under the official PQ formula.
Thus neither the tested pure neural configurations nor this fixed 50/50
ensemble improved the measured development instance-segmentation objective.

{history_table}

The histories describe optimization of training BCE plus soft Dice. They do
not contain validation loss or validation learning curves and cannot establish
overfitting. The measured PQ results support choosing the classical candidate
for this run. They do not identify whether training duration, model capacity,
probability calibration, or conversion of probability maps to instances caused
the neural and hybrid deficits; those factors were not independently isolated.
A full-training fit has no separate generalization score.

## First final submission artifact

The saved [submission CSV]({link('submissions/final_v1.csv')}) contains
**{validation['rows']} rows** for **{validation['images']} images with detections**.
Its recorded local status is `{validation['status']}`, format
`{validation['format']}`. The {len(validation['images_without_detections'])}
images with naturally zero detections are {zero_ids}. No dummy instances were
created to force image coverage.

Kaggle upload status: **`{submission['kaggle_status']}`**. Public and private
scores are unset; there is no official score or platform-acceptance result.
CSV SHA-256: `{submission['sha256']}`.

## Reproduction evidence

The archived [project README]({link('README.md')}) and source belong to the
completed Kaggle v3 artifact at `{artifact_root}`. Canonical evidence is in
[classical results]({link('runs/classical_v1/scores.json')}),
[neural results]({link('runs/unet_v1/scores.json')}),
[final model selection]({link('runs/final_v1/model.json')}), and
[holdout counts]({link('runs/final_v1/holdout_per_image.csv')}). Neural loss
histories and checkpoints are under `runs/unet_v1/fold0/`, `fold1/`, and
`fold2/` inside that archive. Submission metadata, runtimes, and upload status
are under its `submissions/` directory.

Leakage provenance remains in `docs/competition_contract.md`,
`docs/leakage_flag_adjudication.md`, and `data/processed/data_audit.json`.
These results do not establish performance on rare structures, fine barbs,
or a future solar cycle. A same-family provisional review does not provide
independent cross-model validation.
'''
    (output_dir / "results.md").write_text(text)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("results/kaggle_v3/project"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    submission = json.loads((args.artifact_root / "submissions/final_v1.json").read_text())
    evidence = generate_report(args.artifact_root, args.output_dir, submission)
    print(json.dumps({"selected": evidence["selected"], "development_pq": evidence["development_pq"],
                      "locked_holdout": evidence["locked_holdout"],
                      "holdout_month_bootstrap_95_percentile": evidence["holdout_month_bootstrap_95_percentile"],
                      "candidate_count": len(evidence["candidate_scores"]),
                      "kaggle_status": evidence["submission"]["kaggle_status"]}, indent=2))


if __name__ == "__main__":
    main()
