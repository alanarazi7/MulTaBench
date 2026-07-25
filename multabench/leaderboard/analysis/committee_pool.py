"""Analysis (1) of 3 for the rebuttal: sensitivity to the curation MODEL COMMITTEE.

Builds one clean, long-format CSV of raw fold-level scores for the 10 non-end-to-end models
across the full 56-dataset text-tabular pool, all 4 Table-1 conditions -- the single source of
truth this and the other two rebuttal analyses (effect-size threshold, quorum size) will read
from. Does not touch curation_accept.py / model_sensitivity.py / threshold_grid.py /
delta_sweep.py.

Sources (read fresh from the raw per-run CSVs, not the earlier aggregated loaders):
- 5 curation models (LightGBM, CatBoost, TabM, TabPFNv2, TabPFN-2.5), all 4 states:
  tabstar_corpus/*.csv (the original curation-phase run; the same 5 models' rows that also
  appear in text_source/*.csv are ignored, to use a single, unambiguous source per model).
- 5 extra models (RandomForest, RealMLP, TabDPT, TabICLv2, XGBoost):
  - all/ft (Frozen/TAR): text_source/*.csv (36 rejected datasets) + the non-E2E
    more_baselines/*.csv files (20 accepted datasets, short pre-rename names).
  - no_text/text_only: results/sensitivity/fold*.csv (56 datasets, short names for the 20
    originally-accepted ones).

Dataset names are translated to a single canonical (long, post-rename) namespace throughout,
via the same short-to-long resolution used by curation_accept.py (token-subset matching + 2
manual overrides for a typo and a pluralization mismatch).

Run standalone: `python -m multabench.leaderboard.analysis.committee_pool`
"""
import glob
from os.path import dirname, join

import pandas as pd

_RESULTS = join(dirname(__file__), "..", "results")
_CORPUS_CSV = join(_RESULTS, "tabstar_corpus", "text_50_datasets.csv")
_TEXTTABENCH_CSV = join(_RESULTS, "tabstar_corpus", "texttabench_datasets.csv")
_TEXT_SOURCE_DIR = join(_RESULTS, "text_source")
_MORE_BASELINES_DIR = join(_RESULTS, "more_baselines")
_TEXT_DIR = join(_RESULTS, "text")
_SENSITIVITY_DIR = join(_RESULTS, "sensitivity")
_OUT_CSV = join(_RESULTS, "analysis_curation_sensitivity", "pool_scores_long.csv")

CURATION_MODELS = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]
EXTRA_MODELS = ["RandomForest", "RealMLP", "TabDPT", "TabICLv2", "XGBoost"]

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost", "TabM Ⓜ️": "TabM",
    "TabPFN-v2 🤯": "TabPFNv2", "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
    "TabDPT 6️⃣": "TabDPT", "TabICLv2 🗼": "TabICLv2", "RealMLP 🕸": "RealMLP",
    "XGBoost 🌲": "XGBoost", "RandomForest 🌳": "RandomForest",
}

_MORE_BASELINES_NON_E2E_FILES = ["realmlp.csv", "tabdpt.csv", "xgboost.csv", "random_forest.csv", "tabiclv2.csv"]

# The 20 originally-accepted datasets are named differently across sources: tabstar_corpus/
# and text_source/ use the long, post-rename identifiers (e.g.
# "BIN_TEXT_PROFESSIONAL_FAKE_JOB_POSTING"); results/text/, more_baselines/, and
# results/sensitivity/ use the short pre-rename ones (e.g. "BIN_TEXT_FAKE_JOB_POSTING"). Two
# pairs can't be resolved by token matching alone (a typo and a pluralization).
_SHORT_TO_LONG_MANUAL_OVERRIDES = {
    "REG_TEXT_VANCOUVER_SALARIES": "REG_TEXT_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER",
    "REG_TEXT_MONTGOMERY_SALARIES": "REG_TEXT_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY",
}

_COLS = ["model", "state", "dataset", "test_score", "fold"]


def _build_short_to_long_name_map(long_names) -> dict:
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
                              f"(candidates: {candidates})")
        mapping[short] = candidates[0]
    return mapping


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["model"]).copy()
    df["model"] = df["model"].str.strip().map(_MODEL_LABELS)
    df = df[df["model"].notna()]
    df = df.rename(columns={"multimodal_state": "state"})
    return df[_COLS]


def build_long_csv() -> pd.DataFrame:
    """Returns [model, state, dataset, test_score, fold], one row per raw run."""
    # 5 curation models, all 4 states, from tabstar_corpus only.
    df1 = pd.read_csv(_CORPUS_CSV)
    df2 = pd.read_csv(_TEXTTABENCH_CSV)
    if "dataset_name" in df2.columns and "dataset" not in df2.columns:
        df2 = df2.rename(columns={"dataset_name": "dataset"})
    curation_df = _clean(pd.concat([df1, df2], ignore_index=True))
    curation_df = curation_df[curation_df["model"].isin(CURATION_MODELS)]
    long_names = curation_df["dataset"].unique()
    name_map = _build_short_to_long_name_map(long_names)

    # 5 extra models, Frozen/TAR: text_source (36 rejected, long names) + non-E2E
    # more_baselines (20 accepted, short names -> translated).
    ts_frames = [pd.read_csv(f) for f in glob.glob(join(_TEXT_SOURCE_DIR, "*.csv"))]
    text_source_df = _clean(pd.concat(ts_frames, ignore_index=True))
    text_source_df = text_source_df[text_source_df["model"].isin(EXTRA_MODELS)]

    mb_frames = []
    for fname in _MORE_BASELINES_NON_E2E_FILES:
        df = pd.read_csv(join(_MORE_BASELINES_DIR, fname))
        df = df[df["dataset"].str.contains("_TEXT_")].copy()
        df["dataset"] = df["dataset"].map(name_map)
        mb_frames.append(df.dropna(subset=["dataset"]))
    more_baselines_df = _clean(pd.concat(mb_frames, ignore_index=True))

    # 5 extra models, Unimodal (no_text/text_only): results/sensitivity/.
    sens_frames = [pd.read_csv(f) for f in glob.glob(join(_SENSITIVITY_DIR, "*.csv"))]
    sensitivity_df = pd.concat(sens_frames, ignore_index=True)
    sensitivity_df["dataset"] = sensitivity_df["dataset"].map(lambda d: name_map.get(d, d))
    sensitivity_df = _clean(sensitivity_df)

    out = pd.concat([curation_df, text_source_df, more_baselines_df, sensitivity_df], ignore_index=True)
    out = out.drop_duplicates(subset=["model", "dataset", "state", "fold"])
    out["fold"] = out["fold"].astype(int)
    out["test_score"] = out["test_score"].astype(float)
    return out.sort_values(["dataset", "model", "state", "fold"]).reset_index(drop=True)


_STATES = ("no_text", "text_only", "all", "ft")
_FOLDS = range(5)

# The ONE known, genuine gap in the raw source data (a single dropped/failed wandb run),
# confirmed by exhaustively diffing the full expected (model, dataset, state, fold) grid
# against pool_scores_long.csv: exactly this row is missing, nothing else. Hardcoded here so
# it doesn't trip the completeness assertion below; any OTHER missing or duplicate row is
# unexpected and must fail loudly, not be silently averaged over.
_KNOWN_MISSING_ROWS = {("TabPFNv2", "REG_TEXT_CONSUMER_CAR_PRICE_CARDEKHO", "ft", 4)}


def passes(scores: pd.DataFrame, delta: float = 0.001) -> bool:
    """Given ONE model's rows for ONE dataset (exactly 20 rows: 5 folds x 4 states, except
    the one hardcoded gap in _KNOWN_MISSING_ROWS), decide pass/fail.

        Delta_Joint     = mean(all) - max(mean(no_text), mean(text_only))
        Delta_Awareness = mean(ft)  - mean(all)
        passes <=> Delta_Joint > delta AND Delta_Awareness > delta

    Each state's mean is rounded to 3 decimals before differencing (matches the paper's
    reported precision). Asserts (loudly, not a warning) that `scores` is exactly complete
    for one (model, dataset) pair -- no missing rows beyond the one known gap, no
    duplicates/extras -- since silently computing a mean over fewer folds would understate
    variance and bias the pass/fail decision without any visible signal.
    """
    models, datasets = scores["model"].unique(), scores["dataset"].unique()
    assert len(models) == 1 and len(datasets) == 1, (
        f"scores must cover exactly one (model, dataset) pair, "
        f"got models={list(models)} datasets={list(datasets)}"
    )
    model, dataset = models[0], datasets[0]

    expected_rows = {(model, dataset, s, f) for s in _STATES for f in _FOLDS}
    actual_rows = set(zip(scores["model"], scores["dataset"], scores["state"], scores["fold"]))
    missing = expected_rows - actual_rows - _KNOWN_MISSING_ROWS
    extra = actual_rows - expected_rows
    assert not missing, (
        f"Unexpected missing row(s) for ({model}, {dataset}): {sorted(missing)} -- "
        f"this is a NEW data gap, not the one known/hardcoded case. Investigate before "
        f"trusting pass/fail; do not silently add it to _KNOWN_MISSING_ROWS."
    )
    assert not extra, f"Unexpected extra/duplicate row(s) for ({model}, {dataset}): {sorted(extra)}"

    means = scores.groupby("state")["test_score"].mean().round(3)
    delta_joint = means["all"] - max(means["no_text"], means["text_only"])
    delta_awareness = means["ft"] - means["all"]
    return bool(delta_joint > delta and delta_awareness > delta)


if __name__ == "__main__":
    df = build_long_csv()
    df.to_csv(_OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {_OUT_CSV}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"States: {sorted(df['state'].unique())}")
    print(f"Datasets: {df['dataset'].nunique()}")
    counts = df.groupby("model")["dataset"].nunique()
    print("\nDatasets per model:")
    print(counts.to_string())
    expected = df["dataset"].nunique() * 4 * 5
    print(f"\nExpected rows if fully populated (n_datasets * 4 states * 5 folds): {expected}")
