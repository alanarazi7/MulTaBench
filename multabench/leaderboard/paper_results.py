import numpy as np
import pandas as pd
import streamlit as st

from multabench.leaderboard.data.keys import (
    ALL_FEAT, FINETUNED,
    DATASET, FOLD, MODEL, MODE, MM, TEST_SCORE,
    MODALITY, MODALITY_IMAGE, MODALITY_TEXT,
    TASK_TYPE, TASK_REG, TASK_CLS,
)
from multabench.leaderboard.data.loading import load_paper_benchmark_data
from multabench.leaderboard.plots import plot_dataset_performance, plot_ft_vs_all_normalized
from multabench.leaderboard.utils import infer_modality, infer_task_type

_MODE_MAP = {"all": ALL_FEAT, "ft": FINETUNED}
_CONDITIONS = [ALL_FEAT, FINETUNED]
_E2E_IMAGE_MODELS = {"AutoGluon-MM"}
_E2E_TEXT_MODELS  = {"TabSTAR", "ConTextTab", "AutoGluon-MM"}
_ALL_E2E_MODELS   = {"TabSTAR", "ConTextTab", "AutoGluon-MM"}


def _model_win_rate_df(pivot: pd.DataFrame) -> pd.DataFrame:
    stats = pivot.groupby(MODEL)["ft_wins"].agg(["mean", "count"])
    stats["ci"] = 1.96 * np.sqrt(stats["mean"] * (1 - stats["mean"]) / stats["count"]) * 100
    stats["Win %"] = (stats["mean"] * 100).round(1)
    stats["± CI"] = stats["ci"].round(1)
    return stats[["Win %", "± CI"]].sort_values("Win %", ascending=False)


def _display_win_rate(df: pd.DataFrame):
    st.subheader("Win Rate: FT vs Frozen")
    st.caption("Fraction of (model, dataset, fold) triples where FT > Frozen. End-to-end models excluded.")

    df = df.copy()
    df[MODEL] = df[MODEL].apply(lambda x: x.split()[0])
    df = df[~df[MODEL].isin(_ALL_E2E_MODELS)]
    df = df[df[MM].isin(["all", "ft"])]

    pivot = df.pivot_table(index=[MODEL, DATASET, FOLD], columns=MM, values=TEST_SCORE).dropna()
    pivot["ft_wins"] = (pivot["ft"] > pivot["all"]).astype(float)

    for modality, label in [(MODALITY_IMAGE, "Image"), (MODALITY_TEXT, "Text")]:
        mod_pivot = pivot[pivot.index.get_level_values(DATASET).str.contains(f"_{modality.upper()}_")]
        if mod_pivot.empty:
            continue
        total = len(mod_pivot)
        wins = int(mod_pivot["ft_wins"].sum())
        p = wins / total
        ci = 1.96 * np.sqrt(p * (1 - p) / total) * 100
        st.metric(f"{label} overall win rate", f"{p*100:.1f}% ± {ci:.1f}%", f"{wins}/{total} triples")

    st.markdown("**Per model — image + text combined**")
    st.dataframe(_model_win_rate_df(pivot).style.background_gradient(
        subset=["Win %"], cmap="RdYlGn", vmin=0, vmax=100).format("{:.1f}"),
        use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    for col, modality, label in [(c1, MODALITY_IMAGE, "Image"), (c2, MODALITY_TEXT, "Text")]:
        mod_pivot = pivot[pivot.index.get_level_values(DATASET).str.contains(f"_{modality.upper()}_")]
        if mod_pivot.empty:
            continue
        with col:
            st.markdown(f"**{label} — per model**")
            st.dataframe(_model_win_rate_df(mod_pivot).style.background_gradient(
                subset=["Win %"], cmap="RdYlGn", vmin=0, vmax=100).format("{:.1f}"),
                use_container_width=True)

    with st.expander("Per-dataset win rates"):
        c1, c2 = st.columns(2)
        for col, modality, label in [(c1, MODALITY_IMAGE, "Image"), (c2, MODALITY_TEXT, "Text")]:
            mod_pivot = pivot[pivot.index.get_level_values(DATASET).str.contains(f"_{modality.upper()}_")]
            if mod_pivot.empty:
                continue
            with col:
                st.markdown(f"**{label}**")
                stats = mod_pivot.groupby(DATASET)["ft_wins"].agg(["mean", "count"])
                stats["ci"] = (1.96 * np.sqrt(stats["mean"] * (1 - stats["mean"]) / stats["count"]) * 100).round(1)
                stats["Win %"] = (stats["mean"] * 100).round(1)
                stats.index = stats.index.map(lambda x: x.split(f"_{modality.upper()}_", 1)[-1].replace("_", " ").title())
                out = stats[["Win %", "ci"]].rename(columns={"ci": "± CI"}).sort_values("Win %", ascending=False)
                st.dataframe(out.style.background_gradient(subset=["Win %"], cmap="RdYlGn", vmin=0, vmax=100)
                             .format("{:.1f}"), use_container_width=True)


def display_paper_benchmark():
    st.title("🏆 MulTaBench")

    df = load_paper_benchmark_data()
    if df.empty:
        st.warning("No MulTaBench paper benchmark data found.")
        return

    df[MODE] = df[MM].map(_MODE_MAP)
    df[MODALITY] = df[DATASET].apply(infer_modality)
    df[TASK_TYPE] = df[DATASET].apply(infer_task_type)

    task_filter = st.radio("Task type", ["All", TASK_REG, TASK_CLS], horizontal=True, index=0)
    if task_filter != "All":
        df = df[df[TASK_TYPE] == task_filter]

    df_img  = df[df[MODALITY] == MODALITY_IMAGE]
    df_text = df[df[MODALITY] == MODALITY_TEXT]

    st.subheader("Image datasets")
    plot_ft_vs_all_normalized(df_img, native_models=_E2E_IMAGE_MODELS)
    _display_grid(df_img, key="img")
    st.divider()

    st.subheader("Text datasets")
    plot_ft_vs_all_normalized(df_text, native_models=_E2E_TEXT_MODELS)
    _display_grid(df_text, key="txt")
    st.divider()

    st.divider()
    _display_win_rate(df)
    st.divider()

    for dataset in sorted(df[DATASET].unique()):
        _display_dataset(df[df[DATASET] == dataset], dataset)


def _display_grid(df: pd.DataFrame, key: str):
    """Dataset × model grid: chosen mode per (dataset, model, fold), averaged across folds.

    Models that only have one mode always use it regardless of the mode toggle.
    """
    c1, c2 = st.columns(2)
    with c1:
        normalize = st.toggle("Normalize scores (0–1 per row)", key=f"norm_grid_{key}")
    with c2:
        mode_choice = st.radio("Mode", ["ft", "all"], horizontal=True, key=f"mode_grid_{key}")
    preferred_mode = _MODE_MAP[mode_choice]

    # Preferred mode where available; fall back to any available mode for models that lack it
    preferred = df[df[MODE] == preferred_mode].groupby([DATASET, MODEL, FOLD])[TEST_SCORE].mean()
    any_mode = df.groupby([DATASET, MODEL, FOLD])[TEST_SCORE].max()
    per_fold = preferred.combine_first(any_mode).reset_index()

    avg = per_fold.groupby([DATASET, MODEL])[TEST_SCORE].mean().round(3)
    pivot = avg.unstack(MODEL)
    pivot.columns.name = None
    pivot.index.name = None

    # Sort columns by mean score descending
    pivot = pivot[pivot.mean(axis=0).sort_values(ascending=False).index]

    if normalize:
        lo = pivot.min(axis=1)
        hi = pivot.max(axis=1)
        pivot = pivot.subtract(lo, axis=0).divide((hi - lo).clip(lower=1e-9), axis=0).round(2)

    fmt = "{:.2f}" if normalize else "{:.3f}"
    styled = pivot.style.background_gradient(cmap="RdYlGn", axis=1).format(fmt, na_rep="—")
    st.dataframe(styled, use_container_width=True)


def _display_dataset(df: pd.DataFrame, dataset: str):
    st.markdown(f"### {dataset}")
    avg = df.groupby([MODEL, MM])[TEST_SCORE].mean().reset_index()
    pivot = avg.pivot_table(index=MODEL, columns=MM, values=TEST_SCORE).round(3)
    pivot = pivot[[c for c in ["all", "ft"] if c in pivot.columns]]
    pivot.columns = [_MODE_MAP.get(c, c) for c in pivot.columns]
    if ALL_FEAT in pivot.columns:
        pivot = pivot.sort_values(ALL_FEAT, ascending=False)

    plot_df = df.groupby([MODEL, MODE]).agg({TEST_SCORE: "mean"}).reset_index()
    plot_dataset_performance(plot_df, conditions=_CONDITIONS, title="")

    st.dataframe(pivot, use_container_width=True)
    with st.expander("Show fold details"):
        fold_df = df.pivot_table(index=FOLD, columns=[MODEL, MODE], values=TEST_SCORE).round(3)
        st.dataframe(fold_df, use_container_width=True)
    st.divider()
