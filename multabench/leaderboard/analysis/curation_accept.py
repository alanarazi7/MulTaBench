"""Standalone (no streamlit) implementation of the paper's formal curation rule (Appendix A.3).

    Delta_Joint(m)     = S_m(Joint Frozen) - max(S_m(UnimodalStructured), S_m(UnimodalUnstructured))
    Delta_Awareness(m) = S_m(Joint TAR) - S_m(Joint Frozen)
    Accept(D) <=> |{m in M : Delta_Joint(m) > delta AND Delta_Awareness(m) > delta}| >= rho * |M|

Used by the three sensitivity-analysis scripts and by the leaderboard's Sensitivity tab.
"""
import glob
from os.path import dirname, join
from typing import Iterable

import pandas as pd

_RESULTS = join(dirname(__file__), "..", "results")
_CORPUS_CSV = join(_RESULTS, "tabstar_corpus", "text_50_datasets.csv")
_TEXTTABENCH_CSV = join(_RESULTS, "tabstar_corpus", "texttabench_datasets.csv")
_TEXT_SOURCE_DIR = join(_RESULTS, "text_source")
_MORE_BASELINES_DIR = join(_RESULTS, "more_baselines")
_TEXT_DIR = join(_RESULTS, "text")

# more_baselines/*.csv covering the accepted-20 with Frozen/TAR (i.e. non-end-to-end models
# that fit the frozen-embedding-vs-TAR framework). Excludes autogluon_mm.csv, contexttab.csv,
# tabstar.csv -- those are end-to-end models with only a single 'all' condition, no separate
# TAR variant, so they can't contribute a Delta_Awareness signal. tabfm.csv/tabpfnv3.csv were
# removed upstream (models dropped from the panel, not in the paper).
_MORE_BASELINES_NON_E2E_FILES = [
    "realmlp.csv", "tabdpt.csv", "xgboost.csv", "random_forest.csv", "tabiclv2.csv",
]

# The 20 accepted datasets are named differently across CSV sources: tabstar_corpus/ and
# text_source/ use the long, post-rename MulTaBenchDatasetID identifiers (e.g.
# "BIN_TEXT_PROFESSIONAL_FAKE_JOB_POSTING"), while results/text/ and more_baselines/ use the
# original short pre-rename identifiers (e.g. "BIN_TEXT_FAKE_JOB_POSTING"). Two pairs can't be
# resolved by token matching alone (a typo and a pluralization) and are listed explicitly.
_SHORT_TO_LONG_MANUAL_OVERRIDES = {
    "REG_TEXT_VANCOUVER_SALARIES": "REG_TEXT_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER",  # "VANCOUBER" typo in the long name
    "REG_TEXT_MONTGOMERY_SALARIES": "REG_TEXT_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY",  # SALARY vs SALARIES
}

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


def _build_short_to_long_name_map(long_names: Iterable[str]) -> dict:
    """Resolve results/text/*.csv's short (pre-rename) dataset filenames to the long pool
    names used everywhere else, via token-subset matching (short name's tokens, after the
    TASK_MODALITY prefix, must be a subset of the long name's tokens) plus 2 manual overrides
    for a typo and a pluralization mismatch that token matching can't catch. Every short name
    must resolve to exactly one long name -- raises if not, rather than silently mismatching.
    """
    def _tokens(name):
        parts = name.split("_")
        return tuple(parts[:2]), set(parts[2:])

    long_by_prefix = {}
    for name in long_names:
        prefix, toks = _tokens(name)
        long_by_prefix.setdefault(prefix, []).append((name, toks))

    mapping = {}
    for f in glob.glob(join(_TEXT_DIR, "*.csv")):
        short = f.split("/")[-1].replace(".csv", "")
        if short in _SHORT_TO_LONG_MANUAL_OVERRIDES:
            mapping[short] = _SHORT_TO_LONG_MANUAL_OVERRIDES[short]
            continue
        prefix, short_toks = _tokens(short)
        candidates = [name for name, toks in long_by_prefix.get(prefix, []) if short_toks <= toks]
        if len(candidates) != 1:
            raise ValueError(f"Could not uniquely resolve short dataset name {short!r} "
                              f"to a long pool name (candidates: {candidates})")
        mapping[short] = candidates[0]
    return mapping


def load_pool_extended_awareness() -> pd.DataFrame:
    """Per (dataset, model, condition in {JointFrozen, JointTAR}) mean score, for all models
    beyond the 5 curation ones, across the full 56-dataset pool:
    - the 36 REJECTED pool datasets, via text_source/*.csv (12 models, Frozen/TAR only)
    - the 20 ACCEPTED pool datasets, via the non-end-to-end more_baselines/*.csv files
      (_MORE_BASELINES_NON_E2E_FILES), filtered to text-tabular rows and Frozen/TAR only,
      with short dataset names translated to long pool names.
    Folds in the original 5-model pool (Frozen/TAR only) so curation + extra models coexist.
    Delta_Joint is NOT available here for the extra (non-curation) models -- only
    Delta_Awareness, since no unimodal-ablation runs exist for them on the pool.
    """
    base = load_pool_5model()
    base_af = base[base["condition"].isin(["JointFrozen", "JointTAR"])].copy()
    name_map = _build_short_to_long_name_map(base["dataset"].unique())

    all_labels = {**_MODEL_LABELS, **_EXTRA_MODEL_LABELS}

    frames = [pd.read_csv(f) for f in glob.glob(join(_TEXT_SOURCE_DIR, "*.csv"))]

    for fname in _MORE_BASELINES_NON_E2E_FILES:
        df = pd.read_csv(join(_MORE_BASELINES_DIR, fname))
        df = df[df["dataset"].str.contains("_TEXT_")].copy()
        df["dataset"] = df["dataset"].map(name_map)
        df = df.dropna(subset=["dataset"])
        frames.append(df)

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
