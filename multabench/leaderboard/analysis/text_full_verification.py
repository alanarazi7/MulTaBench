"""Verifies MulTaBench-Full admission for the 20 uploaded text-tabular datasets.

Unlike the pool analyses, which read scores obtained from each dataset's ORIGINAL source, this
reads the runs on the uploaded `multabench-full-*` Kaggle artifacts (results/text_full/runs.csv, a
verbatim wandb export -- do not hand-edit it or reconstruct rows, re-export instead). It is
therefore an end-to-end check of the curation recipes AND the upload round-trip, not just of the
datasets' statistical properties: a curation bug that silently dropped or corrupted a text column
would show up here as a lost Joint Signal.

Admission is Joint Signal only (delta = 0.001, quorum 3 of the 5 curation models), so no `ft`
state is needed -- see pass_matrix.compute_joint_delta.

The export carries no `Sweep` column and the runs came from three sources: sweeps `zzlutjez` and
`asuqv7o0`, plus a sharded driver for 57 TabPFN-2.5 cells the second sweep left missing when one
host had no cached copy of the licence-gated checkpoint. Those gaps were not a cartesian product,
which a grid sweep cannot express.

Writes results/analysis_curation_sensitivity/text_full_uploaded_joint_signal.csv.
Run standalone: `python -m multabench.leaderboard.analysis.text_full_verification`
"""
from os.path import dirname, join

import pandas as pd

from multabench.leaderboard.analysis.committee_pool import CURATION_MODELS, _MODEL_LABELS
from multabench.leaderboard.analysis.pass_matrix import DELTA_DEFAULT, compute_joint_delta

_RESULTS = join(dirname(__file__), "..", "results")
_RUNS_CSV = join(_RESULTS, "text_full", "runs.csv")
_OUT_CSV = join(_RESULTS, "analysis_curation_sensitivity", "text_full_uploaded_joint_signal.csv")

# Quorum: a dataset is admitted when a majority of the curation committee sees Joint Signal.
QUORUM = 3


def load_runs() -> pd.DataFrame:
    """The wandb export, normalised to the (model, dataset, state, fold, test_score) schema the
    pass_matrix predicates expect.

    The export carries no `multimodal_state` column, so the state is recovered from the run name,
    which benchmark.py builds as f"{model}_{dataset}_{state}_{fold}". Stripping the known dataset
    and fold leaves the state exactly, which is why this is a parse and not a guess -- the
    assertion below fails loudly rather than silently admitting an unexpected state.
    """
    df = pd.read_csv(_RUNS_CSV)
    assert (df["State"] == "finished").all(), f"Unfinished runs in {_RUNS_CSV}"
    df["state"] = [name.split(f"{ds}_", 1)[-1].rsplit("_", 1)[0]
                   for name, ds in zip(df["Name"], df["dataset"])]
    unexpected = set(df["state"]) - {"no_text", "text_only", "all"}
    assert not unexpected, f"Unparsed state(s) in run names: {sorted(unexpected)}"
    df["model"] = df["model"].map(_MODEL_LABELS)
    unmapped = df["model"].isna().sum()
    assert not unmapped, f"{unmapped} runs have a model label missing from _MODEL_LABELS"
    return df[["model", "dataset", "state", "fold", "test_score"]]


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
