"""Verifies MulTaBench-Full admission for the six uploaded text-tabular classification datasets.

Unlike the pool analyses, which read scores obtained from each dataset's ORIGINAL source, this
reads the runs on the uploaded `multabench-full-*` Kaggle artifacts (results/text_full/). It is
therefore an end-to-end check of the curation recipes AND the upload round-trip, not just of the
datasets' statistical properties: a curation bug that silently dropped or corrupted a text column
would show up here as a lost Joint Signal.

Admission is Joint Signal only (delta = 0.001, quorum 3 of the 5 curation models), so no `ft`
state is needed -- see pass_matrix.compute_joint_delta.

Writes results/analysis_curation_sensitivity/text_full_uploaded_joint_signal.csv.
Run standalone: `python -m multabench.leaderboard.analysis.text_full_verification`
"""
from glob import glob
from os.path import basename, dirname, join, splitext

import pandas as pd

from multabench.leaderboard.analysis.committee_pool import CURATION_MODELS, _MODEL_LABELS
from multabench.leaderboard.analysis.pass_matrix import DELTA_DEFAULT, compute_joint_delta

_RESULTS = join(dirname(__file__), "..", "results")
_RUNS_DIR = join(_RESULTS, "text_full")
_OUT_CSV = join(_RESULTS, "analysis_curation_sensitivity", "text_full_uploaded_joint_signal.csv")

# Quorum: a dataset is admitted when a majority of the curation committee sees Joint Signal.
QUORUM = 3


def load_runs() -> pd.DataFrame:
    """The per-dataset wandb exports, normalised to the (model, dataset, state, fold, test_score)
    schema the pass_matrix predicates expect."""
    frames = []
    for path in sorted(glob(join(_RUNS_DIR, "*.csv"))):
        df = pd.read_csv(path)
        assert (df["State"] == "finished").all(), f"Unfinished runs in {basename(path)}"
        df["dataset"] = splitext(basename(path))[0]
        df["model"] = df["model"].map(_MODEL_LABELS)
        df["state"] = df["multimodal_state"]
        frames.append(df[["model", "dataset", "state", "fold", "test_score"]])
    runs = pd.concat(frames, ignore_index=True)
    unmapped = runs["model"].isna().sum()
    assert not unmapped, f"{unmapped} runs have a model label missing from _MODEL_LABELS"
    return runs


def verdict(runs: pd.DataFrame, delta: float = DELTA_DEFAULT) -> pd.DataFrame:
    """One row per (dataset, model) with Delta_Joint and its pass/fail, plus the dataset verdict."""
    rows = []
    for (dataset, model), scores in runs.groupby(["dataset", "model"]):
        assert model in CURATION_MODELS, f"{model} is not a curation model"
        delta_joint = compute_joint_delta(scores)
        means = scores.groupby("state")["test_score"].mean().round(3)
        rows.append({
            "dataset": dataset, "model": model,
            "no_text": means["no_text"], "text_only": means["text_only"], "all": means["all"],
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
    print(f"{'dataset':<30}{'pass':>7}{'median':>10}{'min':>10}   verdict")
    for dataset, row in summary.iterrows():
        print(f"{dataset:<30}{int(row.joint_pass_5):>5}/5{row.median_delta:>+10.4f}"
              f"{row.min_delta:>+10.4f}   {'ADMIT' if row.admitted else 'REJECT'}")
    print(f"\n{int(summary.admitted.sum())} of {len(summary)} admitted to MulTaBench-Full "
          f"(Joint Signal, delta={DELTA_DEFAULT}, quorum {QUORUM}/{len(CURATION_MODELS)})")


if __name__ == "__main__":
    main()
