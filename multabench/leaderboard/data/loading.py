from os import listdir
from os.path import dirname, join

import pandas as pd
import streamlit as st

from multabench.leaderboard.data.keys import IS_TUNED, MODEL, DATASET, FOLD, MM, E5_MODEL, E5_SMALL, \
    DINO_MODEL, DINO_SMALL, TEST_SCORE


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
