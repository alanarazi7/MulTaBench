import numpy as np
import pandas as pd
import streamlit as st

from multabench.leaderboard.data.keys import DATASET, FOLD, MODEL, MODE, TEST_SCORE, MM
from multabench.leaderboard.data.loading import (
    load_tfidf_comparison_data,
    TFIDF_MODE, E5_ALL_MODE, E5_FT_MODE, IN_BENCHMARK, REJECTION_REASON,
)
from multabench.leaderboard.plots import plot_dataset_performance, plot_normalized_overview
from multabench.leaderboard.text_results import _load_pool_corpus_data, single_variant_name
from multabench.leaderboard.utils import badge

_NO_TABULAR_REASONS = {"[1]", "[1+2]"}


def _is_benchmark(df: pd.DataFrame) -> pd.Series:
    return df[IN_BENCHMARK] == True  # noqa: E712 — explicit for nullable column

def _is_rejected(df: pd.DataFrame) -> pd.Series:
    return df[IN_BENCHMARK] == False  # noqa: E712

def _is_candidate(df: pd.DataFrame) -> pd.Series:
    return df[IN_BENCHMARK].isna()

def _is_no_tabular(df: pd.DataFrame) -> pd.Series:
    return _is_rejected(df) & df[REJECTION_REASON].isin(_NO_TABULAR_REASONS)

def _is_rejected_other(df: pd.DataFrame) -> pd.Series:
    return _is_rejected(df) & ~df[REJECTION_REASON].isin(_NO_TABULAR_REASONS)


def _filter_options(df: pd.DataFrame) -> dict:
    n_all  = df[DATASET].nunique()
    n_in   = df[_is_benchmark(df)][DATASET].nunique()
    n_out  = df[_is_rejected(df)][DATASET].nunique()
    n_cand = df[_is_candidate(df)][DATASET].nunique()
    opts = {
        f"All datasets ({n_all})":                        "all",
        f"In MulTaBench ({n_in})":                        "benchmark",
        f"Accepted — not in benchmark ({n_cand})":        "candidate",
        f"Rejected ({n_out})":                            "rejected",
    }
    return opts


_COMPARISONS = [
    ("ft > frozen",    E5_FT_MODE,  E5_ALL_MODE),
    ("ft > tfidf",     E5_FT_MODE,  TFIDF_MODE),
    ("frozen > tfidf", E5_ALL_MODE, TFIDF_MODE),
]


def _win_rate_row(df: pd.DataFrame) -> dict:
    pivot = df.pivot_table(index=[DATASET, FOLD, MODEL], columns=MODE, values=TEST_SCORE).reset_index()
    row = {}
    for label, winner, loser in _COMPARISONS:
        if winner not in pivot.columns or loser not in pivot.columns:
            row[label] = "—"
            continue
        pair = pivot[[winner, loser]].dropna()
        n = len(pair)
        wins = (pair[winner] > pair[loser]).sum()
        rate = wins / n if n > 0 else float("nan")
        ci = 1.96 * np.sqrt(rate * (1 - rate) / n) if n > 0 else float("nan")
        row[label] = f"{rate*100:.1f}% ± {ci*100:.1f}%"
    return row


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    n_in   = df[_is_benchmark(df)][DATASET].nunique()
    n_cand = df[_is_candidate(df)][DATASET].nunique()
    n_rej  = df[_is_rejected(df)][DATASET].nunique()
    slices = [
        (f"All ({df[DATASET].nunique()})",          df),
        (f"In MulTaBench ({n_in})",                 df[_is_benchmark(df)]),
        (f"Accepted — not in benchmark ({n_cand})", df[_is_candidate(df)]),
        (f"Rejected ({n_rej})",                     df[_is_rejected(df)]),
    ]
    rows = []
    for name, subset in slices:
        row = {"Slice": name}
        row.update(_win_rate_row(subset))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Slice")


def _win_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(index=[DATASET, FOLD, MODEL], columns=MODE, values=TEST_SCORE).reset_index()
    rows = []
    for label, winner, loser in _COMPARISONS:
        if winner not in pivot.columns or loser not in pivot.columns:
            continue
        pair = pivot[[winner, loser]].dropna()
        n = len(pair)
        wins = (pair[winner] > pair[loser]).sum()
        rate = wins / n if n > 0 else float("nan")
        ci = 1.96 * np.sqrt(rate * (1 - rate) / n) if n > 0 else float("nan")
        rows.append({
            "Comparison": label,
            "Pairs": n,
            "Wins": int(wins),
            "Win Rate": f"{rate*100:.1f}%",
            "95% CI": f"± {ci*100:.1f}%",
        })
    return pd.DataFrame(rows).set_index("Comparison")


def _apply_filter(df: pd.DataFrame, filter_val: str) -> pd.DataFrame:
    if filter_val == "benchmark":
        return df[_is_benchmark(df)]
    if filter_val == "rejected":
        return df[_is_rejected(df)]
    if filter_val == "candidate":
        return df[_is_candidate(df)]
    return df  # "all"


_CONDITIONS_POOL = ["Text Only", "No Text", "All", "Finetuned"]
_MM_TO_LABEL = {"all": "All", "ft": "Finetuned", "text_only": "Text Only", "no_text": "No Text"}


def _display_candidate_stats(candidate_datasets: list[str]):
    """Show curation stats (all 4 conditions) for datasets in the candidate bucket."""
    st.subheader("Accepted — not in benchmark: curation stats")
    st.caption(
        "These datasets passed the curation criteria (ft > all, all > no-text, all > text-only "
        "for ≥ 3/5 models) but are not included in the final MulTaBench 20. "
        "Scores from the pool corpus (all 4 conditions)."
    )

    pool = _load_pool_corpus_data()
    pool = pool[pool[DATASET].isin(candidate_datasets)]
    if pool.empty:
        st.warning("No pool corpus data found for candidate datasets.")
        return

    pool[MODE] = pool[MM].map(_MM_TO_LABEL)
    pool = pool[pool[MODE].notna()]

    for dataset in sorted(candidate_datasets):
        df_ds = pool[pool[DATASET] == dataset]
        if df_ds.empty:
            st.markdown(f"### {dataset} — no pool data")
            continue

        # per-model means
        avg = df_ds.groupby([MODEL, MM])[TEST_SCORE].mean().reset_index()
        pivot = avg.pivot_table(index=MODEL, columns=MM, values=TEST_SCORE)

        ft_beats_all    = (pivot.get("ft", pd.Series()) > pivot.get("all", pd.Series())).sum()
        all_beats_notxt = (pivot.get("all", pd.Series()) > pivot.get("no_text", pd.Series())).sum()
        all_beats_txt   = (pivot.get("all", pd.Series()) > pivot.get("text_only", pd.Series())).sum()
        all_three       = (
            (pivot.get("ft", pd.Series()) > pivot.get("all", pd.Series())) &
            (pivot.get("all", pd.Series()) > pivot.get("no_text", pd.Series())) &
            (pivot.get("all", pd.Series()) > pivot.get("text_only", pd.Series()))
        ).sum()
        n_models = len(pivot)

        st.markdown(f"### {dataset}")
        st.caption(
            f"{badge(ft_beats_all)} ft>all: {ft_beats_all}/{n_models}  |  "
            f"{badge(all_beats_notxt)} all>no-text: {all_beats_notxt}/{n_models}  |  "
            f"{badge(all_beats_txt)} all>text-only: {all_beats_txt}/{n_models}  |  "
            f"{badge(all_three)} all three: {all_three}/{n_models}"
        )

        plot_df = df_ds.groupby([MODEL, MODE]).agg({TEST_SCORE: "mean"}).reset_index()
        plot_dataset_performance(plot_df, conditions=_CONDITIONS_POOL, title="", bar_width=0.15)

        fold_df = df_ds.pivot_table(index=FOLD, columns=[MODEL, MODE], values=TEST_SCORE).round(3)
        st.dataframe(fold_df, use_container_width=True)
        st.divider()


def display_tfidf_benchmark():
    st.title("📊 TF-IDF vs E5 Comparison")
    df = load_tfidf_comparison_data()
    if df.empty:
        st.warning("No TF-IDF comparison data found.")
        return

    st.subheader("Win rates by slice")
    st.dataframe(_summary_table(df), use_container_width=True)

    st.divider()
    filter_opts = _filter_options(df)
    filter_label = st.selectbox("Dataset filter", list(filter_opts.keys()), index=0, key="tfidf_filter")
    filter_val = filter_opts[filter_label]
    df_filtered = _apply_filter(df, filter_val)

    n_ds = df_filtered[DATASET].nunique()
    st.caption(f"Comparing TF-IDF, E5-small (frozen), and E5-small (fine-tuned) on {n_ds} text datasets.")

    if filter_val == "candidate":
        candidate_datasets = sorted(df_filtered[DATASET].unique().tolist())
        _display_candidate_stats(candidate_datasets)
        if not df_filtered.empty:
            st.divider()
            st.subheader("TF-IDF results for candidate datasets")
            for dataset in candidate_datasets:
                st.markdown(f"### {dataset}")
                df_ds = df_filtered[df_filtered[DATASET] == dataset]
                plot_df = df_ds.groupby([MODEL, MODE]).agg({TEST_SCORE: "mean"}).reset_index()
                plot_dataset_performance(plot_df, conditions=[TFIDF_MODE], title="", bar_width=0.25)
                fold_df = df_ds.pivot_table(index=FOLD, columns=[MODEL, MODE], values=TEST_SCORE).round(3)
                st.dataframe(fold_df, use_container_width=True)
                st.divider()
        return

    st.subheader("Win rates (across all models and folds)")
    st.dataframe(_win_rate_table(df_filtered), use_container_width=True)

    conditions = [TFIDF_MODE, E5_ALL_MODE, E5_FT_MODE]

    st.subheader("Normalized overview")
    plot_normalized_overview(df_filtered, conditions=conditions, key=f"tfidf_{filter_label}")

    st.divider()

    for dataset in sorted(df_filtered[DATASET].unique()):
        st.markdown(f"### {dataset}")
        df_ds = df_filtered[df_filtered[DATASET] == dataset]
        plot_df = df_ds.groupby([MODEL, MODE]).agg({TEST_SCORE: "mean"}).reset_index()
        plot_dataset_performance(plot_df, conditions=conditions, title="", bar_width=0.15)
        fold_df = df_ds.pivot_table(index=FOLD, columns=[MODEL, MODE], values=TEST_SCORE).round(3)
        st.dataframe(fold_df, use_container_width=True)
        st.divider()
