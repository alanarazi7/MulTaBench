"""Standalone (no streamlit) implementation of the paper's formal curation rule (Appendix A.3).

    Delta_Joint(m)     = S_m(Joint Frozen) - max(S_m(UnimodalStructured), S_m(UnimodalUnstructured))
    Delta_Awareness(m) = S_m(Joint TAR) - S_m(Joint Frozen)
    Accept(D) <=> |{m in M : Delta_Joint(m) > delta AND Delta_Awareness(m) > delta}| >= rho * |M|

Used by the three sensitivity-analysis scripts and by the leaderboard's Sensitivity tab.
"""
from os.path import dirname, join
from typing import Iterable

import pandas as pd

_RESULTS = join(dirname(__file__), "..", "results")
_CORPUS_CSV = join(_RESULTS, "tabstar_corpus", "text_50_datasets.csv")
_TEXTTABENCH_CSV = join(_RESULTS, "tabstar_corpus", "texttabench_datasets.csv")
_TEXT_SOURCE_DIR = join(_RESULTS, "text_source")
_MORE_BASELINES_DIR = join(_RESULTS, "more_baselines")

# The 5 curation learners, in paper order.
CURATION_MODELS = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM",
    "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM",
    "TabPFN-v2 🤯": "TabPFNv2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
}
# Extra models available (Frozen/TAR only) via text_source/ + more_baselines/.
_EXTRA_MODEL_LABELS = {
    "TabDPT 6️⃣": "TabDPT",
    "TabICLv2 🗼": "TabICLv2",
    "TabFM 🇬": "TabFM",
    "TabPFN-v3 🆕": "TabPFN-v3",
    "RealMLP 🕸": "RealMLP",
    "XGBoost 🌲": "XGBoost",
    "RandomForest 🌳": "RandomForest",
}

_CONDITION_MAP = {"no_text": "UnimodalStructured", "text_only": "UnimodalUnstructured",
                   "all": "JointFrozen", "ft": "JointTAR"}
_AWARENESS_ONLY_MAP = {"all": "JointFrozen", "ft": "JointTAR"}

DELTA_DEFAULT = 0.001
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


def load_pool_extended_awareness() -> pd.DataFrame:
    """Per (dataset, model, condition in {JointFrozen, JointTAR}) mean score, across the
    12-model text_source (36 rejected pool datasets) + more_baselines (20 accepted) +
    the original 5-model pool restricted to Frozen/TAR. Delta_Joint is NOT available here for
    the extra (non-curation) models -- only Delta_Awareness.
    """
    import glob
    frames = []
    for f in glob.glob(join(_TEXT_SOURCE_DIR, "*.csv")):
        frames.append(pd.read_csv(f))
    for f in glob.glob(join(_MORE_BASELINES_DIR, "*.csv")):
        frames.append(pd.read_csv(f))
    # Also fold in the original 5-model pool (all/ft only) so extra + curation models coexist.
    base = load_pool_5model()
    base_af = base[base["condition"].isin(["JointFrozen", "JointTAR"])].copy()

    all_labels = {**_MODEL_LABELS, **_EXTRA_MODEL_LABELS}
    extra = pd.concat(frames, ignore_index=True)
    extra = extra.dropna(subset=["model"])
    extra["model"] = extra["model"].str.strip().map(all_labels)
    extra = extra[extra["model"].notna()]
    extra["condition"] = extra["multimodal_state"].map(_AWARENESS_ONLY_MAP)
    extra = extra[extra["condition"].notna()]
    extra = (extra.groupby(["dataset", "model", "condition"])["test_score"]
                   .mean().reset_index().rename(columns={"test_score": "score"}))

    combined = pd.concat([base_af, extra], ignore_index=True)
    # A (dataset, model) pair may appear in both the base pool and text_source/more_baselines
    # for the 5 curation models -- keep one (they're the same underlying runs).
    combined = combined.drop_duplicates(subset=["dataset", "model", "condition"])
    return combined


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


def compute_deltas_awareness_only(df: pd.DataFrame) -> pd.DataFrame:
    """Like compute_deltas, but only delta_awareness (for models lacking unimodal ablations)."""
    pivot = df.pivot_table(index=["dataset", "model"], columns="condition", values="score").round(3)
    out = pivot.reset_index()
    out["delta_awareness"] = out["JointTAR"] - out["JointFrozen"]
    return out[["dataset", "model", "delta_awareness"]]


def per_model_pass(deltas: pd.DataFrame, delta: float = DELTA_DEFAULT) -> pd.DataFrame:
    """[dataset, model, passes] -- passes iff both deltas exceed delta."""
    out = deltas.copy()
    out["passes"] = (out["delta_joint"] > delta) & (out["delta_awareness"] > delta)
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
