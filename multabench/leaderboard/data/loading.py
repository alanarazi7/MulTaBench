from os import listdir
from os.path import dirname, isdir, join

import pandas as pd
import streamlit as st

from multabench.leaderboard.data.keys import IS_TUNED, MODEL, DATASET, FOLD, MM, MODE, E5_MODEL, E5_SMALL, \
    DINO_MODEL, DINO_SMALL, TEST_SCORE, PCA_COMPONENTS


def _load_csv(f: str) -> pd.DataFrame:
    try:
        return pd.read_csv(f)
    except pd.errors.ParserError as pe:
        raise ValueError(f"Failed to load {f}, error: {pe}")


def _require_columns(df: pd.DataFrame, required: list[str], filepath: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"File {filepath!r} is missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )


def _load_result_dir(result_dir: str, required_cols: list[str]) -> list[pd.DataFrame]:
    dfs = []
    for f in [join(result_dir, f) for f in listdir(result_dir) if f.endswith('.csv')]:
        df = _load_csv(f)
        _require_columns(df, required_cols, f)
        df = df.dropna(subset=[MODEL])
        df[MODEL] = df[MODEL].apply(lambda x: x.strip())
        df[IS_TUNED] = df[MODEL].apply(lambda x: 'Tuned' in x)
        if E5_MODEL not in df.columns:
            df[E5_MODEL] = E5_SMALL
        if DINO_MODEL not in df.columns:
            df[DINO_MODEL] = DINO_SMALL
        if TEST_SCORE in df.columns:
            df[TEST_SCORE] = df[TEST_SCORE].clip(lower=-0.1)
        dfs.append(df)
    return dfs


@st.cache_data
def load_multimodal_leaderboard_data() -> pd.DataFrame:
    try:
        return get_results_data()
    except ValueError as ve:
        st.error(f"Error loading results data: {ve}")
        raise ve


def get_results_data() -> pd.DataFrame:
    result_dir = join(dirname(__file__), '..', 'results', 'text')
    return pd.concat(_load_result_dir(result_dir, required_cols=[FOLD, MODEL, DATASET]))


@st.cache_data
def load_large_results_data() -> pd.DataFrame:
    results_root = join(dirname(__file__), '..', 'results')
    dfs = []
    for subdir in ('images_large', 'text_large'):
        dfs += _load_result_dir(join(results_root, subdir), required_cols=[FOLD, MODEL, DATASET])
    return pd.concat(dfs) if dfs else pd.DataFrame()


@st.cache_data
def load_multabench_data() -> pd.DataFrame:
    results_root = join(dirname(__file__), '..', 'results')
    dfs = []
    for subdir in ('images', 'text'):
        dfs += _load_result_dir(join(results_root, subdir), required_cols=[MODEL, DATASET, FOLD, MM, "test_score"])
    if not dfs:
        return pd.DataFrame()
    return pd.concat([df[[MODEL, DATASET, FOLD, MM, "test_score"]] for df in dfs])


@st.cache_data
def load_triple_data() -> pd.DataFrame:
    """Load combined image + tabular_image_text results for the Triple panel (8 states)."""
    required = [MODEL, DATASET, FOLD, MM, "test_score"]
    results_root = join(dirname(__file__), '..', 'results')

    # Images directory uses the standard 'dataset' column
    images_dfs = _load_result_dir(join(results_root, 'images'), required_cols=required)

    # tabular_image_text uses 'dataset_name' instead of 'dataset'
    txt_img_dfs = []
    txt_img_dir = join(results_root, 'tabular_image_text')
    if isdir(txt_img_dir):
        for fname in [join(txt_img_dir, fn) for fn in listdir(txt_img_dir) if fn.endswith('.csv')]:
            df = _load_csv(fname)
            if 'dataset_name' in df.columns and 'dataset' not in df.columns:
                df = df.rename(columns={'dataset_name': 'dataset'})
            _require_columns(df, required, fname)
            df = df.dropna(subset=[MODEL])
            df[MODEL] = df[MODEL].apply(lambda x: x.split()[0])
            df[IS_TUNED] = df[MODEL].apply(lambda x: 'Tuned' in x)
            if E5_MODEL not in df.columns:
                df[E5_MODEL] = E5_SMALL
            if DINO_MODEL not in df.columns:
                df[DINO_MODEL] = DINO_SMALL
            if "test_score" in df.columns:
                df["test_score"] = df["test_score"].clip(lower=0)
            txt_img_dfs.append(df)

    all_dfs = images_dfs + txt_img_dfs
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat([df[[MODEL, DATASET, FOLD, MM, "test_score"]] for df in all_dfs])


@st.cache_data
def load_paper_benchmark_data() -> pd.DataFrame:
    """Load all small-encoder (all/ft only) results for the MulTaBench paper tab.

    Combines the curated images/text results with every CSV in more_baselines/.
    Adding a new model is as simple as dropping a new CSV into more_baselines/.
    Rows are filtered to all/ft states and dino-small / e5-small only.
    """
    required = [MODEL, DATASET, FOLD, MM, "test_score"]
    results_root = join(dirname(__file__), '..', 'results')
    dfs = []
    for subdir in ('images', 'text', 'more_baselines'):
        dfs += _load_result_dir(join(results_root, subdir), required_cols=required)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat([d[[MODEL, DATASET, FOLD, MM, "test_score", E5_MODEL, DINO_MODEL]] for d in dfs])
    df = df[df[MM].isin(["all", "ft"])]
    df = df[(df[E5_MODEL] == E5_SMALL) & (df[DINO_MODEL] == DINO_SMALL)]
    return df.reset_index(drop=True)


@st.cache_data
def load_pca_comparison_data() -> pd.DataFrame:
    """Load PCA-dimension comparison data (15 / 30-default / 60).

    Sources:
    - analysis_pca/{ft,all}_{15,60}.csv  →  pca_components in file
    - images/ + text/ curation dirs      →  pca_components = 30 (default)

    Only ft and all multimodal states are kept.
    Returns columns: model, dataset, fold, multimodal_state, test_score, pca_components.
    """
    results_root = join(dirname(__file__), '..', 'results')
    dfs = []
    cols = [MODEL, DATASET, FOLD, MM, TEST_SCORE, PCA_COMPONENTS]

    # 1. PCA experiment files (have pca_components column, dataset_name key)
    pca_dir = join(results_root, 'analysis_pca')
    for fname in [f for f in listdir(pca_dir) if f.endswith('.csv')]:
        df = _load_csv(join(pca_dir, fname))
        if 'no_pca' in df.columns:
            continue  # handled separately by load_no_pca_data
        if 'dataset_name' in df.columns and DATASET not in df.columns:
            df = df.rename(columns={'dataset_name': DATASET})
        df = df.dropna(subset=[MODEL])
        df[MODEL] = df[MODEL].str.strip()
        df[IS_TUNED] = df[MODEL].apply(lambda x: 'Tuned' in x)
        df[TEST_SCORE] = df[TEST_SCORE].clip(lower=-0.1)
        df = df[df[MM].isin(["ft", "all"])]
        dfs.append(df[cols])

    # 2. Curation data: images/ and text/ directories (pca_components = 30 default)
    for subdir in ('images', 'text'):
        for raw_df in _load_result_dir(join(results_root, subdir), required_cols=[MODEL, DATASET, FOLD, MM, TEST_SCORE]):
            df = raw_df[raw_df[MM].isin(["ft", "all"])].copy()
            df[PCA_COMPONENTS] = 30
            dfs.append(df[cols])

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


@st.cache_data
def load_no_pca_data() -> pd.DataFrame:
    """Load no-PCA runs (no_pca=yes) from analysis_pca/."""
    results_root = join(dirname(__file__), '..', 'results')
    pca_dir = join(results_root, 'analysis_pca')
    cols = [MODEL, DATASET, FOLD, MM, TEST_SCORE]
    dfs = []
    for fname in listdir(pca_dir):
        if not fname.endswith('.csv'):
            continue
        df = _load_csv(join(pca_dir, fname))
        if 'no_pca' not in df.columns:
            continue
        df = df[df['no_pca'].astype(str).str.lower().isin(['yes', 'true'])]
        if df.empty:
            continue
        df = df.dropna(subset=[MODEL])
        df[MODEL] = df[MODEL].str.strip()
        df[TEST_SCORE] = df[TEST_SCORE].clip(lower=-0.1)
        df = df[df[MM].isin(['all', 'ft'])]
        dfs.append(df[cols])
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


TFIDF_MODE = "TF-IDF"
E5_ALL_MODE = "E5 (frozen)"
E5_FT_MODE  = "E5 (fine-tuned)"


def _extract_dataset_from_name(name: str) -> str:
    """Parse dataset ID from a run Name like 'tabm_MUL_TEXT_MICHELIN_RESTAURANTS_4'."""
    parts = name.split("_")
    return "_".join(parts[1:-1])


IN_BENCHMARK = "in_benchmark"
REJECTION_REASON = "rejection_reason"


def _compute_rejection_reasons(results_root: str) -> dict[str, str]:
    """Return {dataset_name: reason} for all datasets in the pool corpus.

    reason is one of: '[1]', '[2]', '[1+2]', '[3]', '[3*]'
    Mirrors the logic in text_results._rejection_reason / _compute_curation_summary.
    """
    corpus_root = join(results_root, 'tabstar_corpus')
    dfs = []
    for fname in ['text_50_datasets.csv', 'texttabench_datasets.csv']:
        cdf = _load_csv(join(corpus_root, fname))
        if 'dataset_name' in cdf.columns and DATASET not in cdf.columns:
            cdf = cdf.rename(columns={'dataset_name': DATASET})
        dfs.append(cdf)
    corpus = pd.concat(dfs, ignore_index=True)
    corpus[MODEL] = corpus[MODEL].str.strip()
    corpus[TEST_SCORE] = corpus[TEST_SCORE].clip(lower=-0.1)

    avg = corpus.groupby([DATASET, MODEL, MM])[TEST_SCORE].mean().reset_index()
    pivot = avg.pivot_table(index=[DATASET, MODEL], columns=MM, values=TEST_SCORE).reset_index()

    reasons = {}
    for dataset, grp in pivot.groupby(DATASET):
        def _mean_col(col):
            return grp[col].mean() if col in grp.columns else float("nan")
        text = _mean_col("text_only")
        no_text = _mean_col("no_text")
        all_ = _mean_col("all")
        r1 = pd.notna(text)    and pd.notna(all_) and text    >= all_
        r2 = pd.notna(no_text) and pd.notna(all_) and no_text >= all_
        if r1 and r2:
            reasons[dataset] = "[1+2]"
        elif r1:
            reasons[dataset] = "[1]"
        elif r2:
            reasons[dataset] = "[2]"
        else:
            reasons[dataset] = "[3+]"  # [3] or [3*] — fine-tuning / inconsistency issues
    return reasons

@st.cache_data
def load_tfidf_comparison_data() -> pd.DataFrame:
    """Load TF-IDF vs E5 frozen/ft for all datasets that have TF-IDF results.

    Returns a dataframe with an IN_BENCHMARK column (nullable):
      True  — confirmed in the final MulTaBench 20 text-tabular datasets (tfidf.csv)
      False — evaluated and rejected during curation
      pd.NA — in ACCEPTED_TEXT_DATASETS but not yet confirmed in the curated 20
    """
    from multabench.datasets.text_benchmarks import ACCEPTED_TEXT_DATASETS
    accepted_names = {d.name for d in ACCEPTED_TEXT_DATASETS}

    results_root = join(dirname(__file__), '..', 'results')
    cols = [MODEL, DATASET, FOLD, MODE, TEST_SCORE, IN_BENCHMARK]
    dfs = []

    # ── TF-IDF: confirmed in-benchmark datasets (tfidf.csv) ──
    tfidf_path = join(results_root, 'analysis_tfidf', 'tfidf.csv')
    tfidf_df = _load_csv(tfidf_path)
    tfidf_df = tfidf_df.dropna(subset=[MODEL])
    tfidf_df[MODEL] = tfidf_df[MODEL].str.strip()
    tfidf_df[DATASET] = tfidf_df["Name"].apply(_extract_dataset_from_name)
    tfidf_df[TEST_SCORE] = tfidf_df[TEST_SCORE].clip(lower=-0.1)
    tfidf_df[MODE] = TFIDF_MODE
    tfidf_df[IN_BENCHMARK] = True
    confirmed_benchmark = set(tfidf_df[DATASET].unique())
    dfs.append(tfidf_df[cols])

    # ── TF-IDF: other datasets (tfidf_other_baselines*.csv) ──
    # True if confirmed, pd.NA if accepted-but-not-confirmed, False if rejected.
    tfidf_other_parts = []
    for fname in ('tfidf_other_baselines.csv', 'tfidf_other_baselines_new.csv'):
        fpath = join(results_root, 'analysis_tfidf', fname)
        part = _load_csv(fpath)
        part = part.dropna(subset=[MODEL])
        part[MODEL] = part[MODEL].str.strip()
        if 'dataset_name' in part.columns and DATASET not in part.columns:
            part = part.rename(columns={'dataset_name': DATASET})
        part[TEST_SCORE] = part[TEST_SCORE].clip(lower=-0.1)
        part[MODE] = TFIDF_MODE
        part[IN_BENCHMARK] = part[DATASET].apply(
            lambda d: True if d in confirmed_benchmark
            else (pd.NA if d in accepted_names else False)
        )
        tfidf_other_parts.append(part[[MODEL, DATASET, FOLD, MODE, TEST_SCORE, IN_BENCHMARK]])
    tfidf_other = pd.concat(tfidf_other_parts, ignore_index=True)
    dfs.append(tfidf_other)

    # ── E5 frozen/ft: confirmed in-benchmark datasets from text/ directory ──
    mm_map = {"all": E5_ALL_MODE, "ft": E5_FT_MODE}
    text_raw = _load_result_dir(join(results_root, 'text'), required_cols=[MODEL, DATASET, FOLD, MM, TEST_SCORE])
    for df in text_raw:
        df = df[(df[MM].isin(["all", "ft"])) & (df[E5_MODEL] == E5_SMALL)].copy()
        df[MODE] = df[MM].map(mm_map)
        df = df[df[MODE].notna()]
        df[IN_BENCHMARK] = df[DATASET].apply(
            lambda d: True if d in confirmed_benchmark else pd.NA
        )
        dfs.append(df[[MODEL, DATASET, FOLD, MODE, TEST_SCORE, IN_BENCHMARK]])

    # ── E5 frozen/ft: rejected datasets from tabstar_corpus pool CSVs ──
    # Only load datasets confirmed rejected (IN_BENCHMARK == False).
    rejected_tfidf_datasets = set(
        tfidf_other.loc[tfidf_other[IN_BENCHMARK] == False, DATASET].unique()
    )
    corpus_root = join(results_root, 'tabstar_corpus')
    corpus_dfs = []
    for fname in ['text_50_datasets.csv', 'texttabench_datasets.csv']:
        fpath = join(corpus_root, fname)
        cdf = _load_csv(fpath)
        if 'dataset_name' in cdf.columns and DATASET not in cdf.columns:
            cdf = cdf.rename(columns={'dataset_name': DATASET})
        corpus_dfs.append(cdf)
    corpus = pd.concat(corpus_dfs, ignore_index=True)
    corpus[MODEL] = corpus[MODEL].str.strip()
    corpus[TEST_SCORE] = corpus[TEST_SCORE].clip(lower=-0.1)
    corpus = corpus[corpus[MM].isin(["all", "ft"])]
    corpus = corpus[corpus[DATASET].isin(rejected_tfidf_datasets)]
    corpus[MODE] = corpus[MM].map(mm_map)
    corpus[IN_BENCHMARK] = False
    dfs.append(corpus[[MODEL, DATASET, FOLD, MODE, TEST_SCORE, IN_BENCHMARK]])

    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)

    # Attach rejection reason for rejected datasets (computed from pool corpus conditions).
    reasons = _compute_rejection_reasons(results_root)
    result[REJECTION_REASON] = result[DATASET].map(reasons).where(result[IN_BENCHMARK] == False)
    return result
