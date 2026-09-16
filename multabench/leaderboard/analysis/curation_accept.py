"""Standalone (no streamlit) implementation of the paper's formal curation rule (Appendix A.3).

    Delta_Joint(m)     = S_m(Joint Frozen) - max(S_m(UnimodalStructured), S_m(UnimodalUnstructured))
    Delta_Awareness(m) = S_m(Joint TAR) - S_m(Joint Frozen)
    Accept(D) <=> |{m in M : Delta_Joint(m) >= delta AND Delta_Awareness(m) >= delta}| >= rho * |M|

Used by the threshold-sensitivity scripts and by the leaderboard's Sensitivity tab.
"""
from os.path import dirname, join
from typing import Iterable

import pandas as pd

from multabench.leaderboard.analysis.pass_matrix import DELTA_DEFAULT, passes_delta

_RESULTS = join(dirname(__file__), "..", "results")
_CORPUS_CSV = join(_RESULTS, "tabstar_corpus", "text_50_datasets.csv")
_TEXTTABENCH_CSV = join(_RESULTS, "tabstar_corpus", "texttabench_datasets.csv")

# The 5 curation learners, in paper order.
CURATION_MODELS = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM",
    "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM",
    "TabPFN-v2 🤯": "TabPFNv2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
}

_CONDITION_MAP = {"no_text": "UnimodalStructured", "text_only": "UnimodalUnstructured",
                   "all": "JointFrozen", "ft": "JointTAR"}

RHO_DEFAULT = 3 / 5


def load_pool_5model() -> pd.DataFrame:
    """Per (dataset, model, condition) mean test_score over folds, for the 56-dataset pool,
    the 5 curation learners, and all 4 conditions."""
    df1 = pd.read_csv(_CORPUS_CSV)
    df2 = pd.read_csv(_TEXTTABENCH_CSV)
    if "dataset_name" in df2.columns and "dataset" not in df2.columns:
        df2 = df2.rename(columns={"dataset_name": "dataset"})
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna(subset=["model"])
    df["model"] = df["model"].str.strip().map(_MODEL_LABELS)
    df = df[df["model"].notna()]
    df["condition"] = df["multimodal_state"].map(_CONDITION_MAP)
    df = df[df["condition"].notna()]
    return (df.groupby(["dataset", "model", "condition"])["test_score"]
              .mean().reset_index().rename(columns={"test_score": "score"}))


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """df: [dataset, model, condition, score] with condition in the 4 Table-1 names.
    Returns [dataset, model, delta_joint, delta_awareness], rounding each condition's mean
    score to 3 decimals before differencing (matches the repo's existing rounding convention).
    """
    pivot = df.pivot_table(index=["dataset", "model"], columns="condition", values="score").round(3)
    for col in ["UnimodalStructured", "UnimodalUnstructured", "JointFrozen", "JointTAR"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")
    out = pivot.reset_index()
    out["delta_joint"] = out["JointFrozen"] - out[["UnimodalStructured", "UnimodalUnstructured"]].max(axis=1)
    out["delta_awareness"] = out["JointTAR"] - out["JointFrozen"]
    return out[["dataset", "model", "delta_joint", "delta_awareness"]]


def per_model_pass(deltas: pd.DataFrame, delta: float = DELTA_DEFAULT) -> pd.DataFrame:
    """[dataset, model, passes] -- passes iff both deltas reach delta."""
    out = deltas.copy()
    out["passes"] = [passes_delta(j, delta) and passes_delta(a, delta)
                     for j, a in zip(out["delta_joint"], out["delta_awareness"])]
    return out[["dataset", "model", "passes"]]


def accept_set(deltas: pd.DataFrame, models: Iterable[str], delta: float = DELTA_DEFAULT,
               rho: float = RHO_DEFAULT) -> dict:
    """Accept(D) <=> count(passing models in `models`) >= rho * |models|.
    Returns {dataset: bool}.
    """
    models = list(models)
    sub = deltas[deltas["model"].isin(models)]
    votes = per_model_pass(sub, delta=delta)
    counts = votes.groupby("dataset")["passes"].sum()
    threshold = rho * len(models)
    # A dataset with no rows for these models (shouldn't happen for the 56-pool + 5 models) gets 0.
    all_datasets = deltas["dataset"].unique()
    return {d: bool(counts.get(d, 0) >= threshold) for d in all_datasets}
