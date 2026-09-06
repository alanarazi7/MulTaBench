"""Verifies MulTaBench-Full admission for the uploaded image-tabular datasets.

The image counterpart of text_full_verification.py. It reads runs on the uploaded
`multabench-full-*` Kaggle artifacts (results/image_full/runs.csv, a verbatim wandb export -- do
not hand-edit it or reconstruct rows, re-export instead), so it checks the curation recipes and
the upload round-trip together, not just the datasets' statistical properties.

Admission is Joint Signal only (delta = 0.001, quorum 3 of the 5 curation models), so no `ft`
state is needed -- see pass_matrix.compute_joint_delta.

Writes results/analysis_curation_sensitivity/image_full_uploaded_joint_signal.csv.
Run standalone: `python -m multabench.leaderboard.analysis.image_full_verification`
"""
from os.path import dirname, join

import pandas as pd

from multabench.leaderboard.analysis.committee_pool import CURATION_MODELS, _MODEL_LABELS
from multabench.leaderboard.analysis.pass_matrix import (
    DELTA_DEFAULT, IMAGE_JOINT_STATES, compute_joint_delta,
)

_RESULTS = join(dirname(__file__), "..", "results")
_RUNS_CSV = join(_RESULTS, "image_full", "runs.csv")
_OUT_CSV = join(_RESULTS, "analysis_curation_sensitivity",
                "image_full_uploaded_joint_signal.csv")

QUORUM = 3

JOINT_SIGNAL_STATES = {"non", "img", "all"}


def load_runs() -> pd.DataFrame:
    """The wandb export, normalised to (model, dataset, state, fold, test_score)."""
    df = pd.read_csv(_RUNS_CSV)
    assert (df["State"] == "finished").all(), f"Unfinished runs in {_RUNS_CSV}"

    # wandb records the CLI argument, so `ft` stays distinguishable from `all`.
    df["state"] = df["multimodal_state"]
    extra = set(df["state"].unique()) - JOINT_SIGNAL_STATES
    assert not extra, (f"Export carries state(s) {sorted(extra)}; Joint Signal is measured on the "
                       f"frozen non/img/all conditions only.")

    df["model"] = df["model"].map(_MODEL_LABELS)
    unmapped_models = df["model"].isna().sum()
    assert not unmapped_models, f"{unmapped_models} runs have a model label missing from _MODEL_LABELS"
    return df[["model", "dataset", "state", "fold", "test_score"]]


def verdict(runs: pd.DataFrame, delta: float = DELTA_DEFAULT) -> pd.DataFrame:
    """One row per (dataset, model) with Delta_Joint and its pass/fail, plus the dataset verdict."""
    rows = []
    for (dataset, model), scores in runs.groupby(["dataset", "model"]):
        assert model in CURATION_MODELS, f"{model} is not a curation model"
        delta_joint = compute_joint_delta(scores, states=IMAGE_JOINT_STATES)
        means = scores.groupby("state")["test_score"].mean().round(3)
        rows.append({
            "dataset": dataset, "model": model,
            "non": means["non"], "img": means["img"], "all": means["all"],
            "delta_joint": round(delta_joint, 4), "joint_pass": delta_joint > delta,
        })
    df = pd.DataFrame(rows)
    per_dataset = df.groupby("dataset")["joint_pass"].sum().rename("joint_pass_5")
    df = df.merge(per_dataset, on="dataset")
    df["admitted"] = df["joint_pass_5"] >= QUORUM
    return df.sort_values(["joint_pass_5", "dataset", "model"], ascending=[False, True, True])


def main() -> None:
    runs = load_runs()
    print(f"Loaded {len(runs)} runs: {runs['dataset'].nunique()} datasets x "
          f"{runs['model'].nunique()} models x {runs['state'].nunique()} states x "
          f"{runs['fold'].nunique()} folds")

    df = verdict(runs)
    df.to_csv(_OUT_CSV, index=False)
    print(f"Wrote {len(df)} (dataset, model) rows to {_OUT_CSV}\n")

    summary = (df.groupby("dataset")
                 .agg(joint_pass_5=("joint_pass_5", "first"),
                      median_delta=("delta_joint", "median"),
                      min_delta=("delta_joint", "min"),
                      admitted=("admitted", "first"))
                 .sort_values(["joint_pass_5", "median_delta"], ascending=False))
    print(f"{'dataset':<34}{'pass':>7}{'median':>10}{'min':>10}   verdict")
    for dataset, row in summary.iterrows():
        print(f"{dataset:<34}{int(row.joint_pass_5):>5}/5{row.median_delta:>+10.4f}"
              f"{row.min_delta:>+10.4f}   {'ADMIT' if row.admitted else 'REJECT'}")
    print(f"\n{int(summary.admitted.sum())} of {len(summary)} admitted to MulTaBench-Full "
          f"(Joint Signal, delta={DELTA_DEFAULT}, quorum {QUORUM}/{len(CURATION_MODELS)})")


if __name__ == "__main__":
    main()
