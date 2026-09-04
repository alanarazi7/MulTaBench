import pandas as pd
import streamlit as st

from multabench.leaderboard.data.keys import (
    ALL_FEAT, FINETUNED, IMG_ONLY, NO_IMG, TEXT_ONLY, NO_TEXT,
    DATASET, FOLD, MODEL, MODE, MM, TEST_SCORE,
    MODALITY, MODALITY_IMAGE, MODALITY_TEXT,
)
from multabench.datasets.all_multabench_datasets import MULTABENCH_CORE_IMAGE
from multabench.leaderboard.data.loading import load_multabench_data
from multabench.leaderboard.plots import plot_dataset_performance, plot_ft_vs_all_normalized
from multabench.leaderboard.utils import infer_modality, badge

_CORE_IMAGE_NAMES = {d.name for d in MULTABENCH_CORE_IMAGE}

_MODE_MAP = {
    "all":       ALL_FEAT,
    "ft":        FINETUNED,
    "img":       IMG_ONLY,
    "non":       NO_IMG,
    "text_only": TEXT_ONLY,
    "no_text":   NO_TEXT,
}

_IMAGE_CONDITIONS = [IMG_ONLY, NO_IMG, ALL_FEAT, FINETUNED]
_TEXT_CONDITIONS  = [TEXT_ONLY, NO_TEXT, ALL_FEAT, FINETUNED]



_NATIVE_TEXT_MODELS = {"ConTextTab", "TabSTAR"}


def display_multabench():
    st.title("🧹 Curation")

    df = load_multabench_data()
    if df.empty:
        st.warning("No MulTaBench data found.")
        return

    df[MODE] = df[MM].map(_MODE_MAP)
    df = df[df[MODE].notna()]
    df[MODALITY] = df[DATASET].apply(infer_modality)

    summary = _compute_summary(df)
    _display_viewer(df, summary)


def _compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    avg = df.groupby([DATASET, MODEL, MM])[TEST_SCORE].mean().reset_index()
    avg[TEST_SCORE] = avg[TEST_SCORE].round(3)
    pivot = avg.pivot_table(index=[DATASET, MODEL], columns=MM, values=TEST_SCORE).reset_index()

    # Per-row comparison: pick the first candidate with non-NaN values for that row.
    # This handles mixed image+text data where both "non"/"no_text" columns exist but
    # only one is populated per dataset.
    def _compare_row(row, candidates):
        all_val = row.get("all", float("nan"))
        for c in candidates:
            c_val = row.get(c, float("nan"))
            if pd.notna(all_val) and pd.notna(c_val):
                return all_val > c_val
        return False

    pivot["all>non"] = pivot.apply(lambda r: _compare_row(r, ["non", "no_text"]), axis=1)
    pivot["all>img"] = pivot.apply(lambda r: _compare_row(r, ["img", "text_only"]), axis=1)
    pivot["ft>all"] = pivot.apply(
        lambda r: r.get("ft", float("nan")) > r.get("all", float("nan"))
        if pd.notna(r.get("ft", float("nan"))) and pd.notna(r.get("all", float("nan"))) else False,
        axis=1,
    )
    pivot["all_three"] = pivot["ft>all"] & pivot["all>non"] & pivot["all>img"]

    counts = pivot.groupby(DATASET)[["ft>all", "all>non", "all>img", "all_three"]].sum().astype(int)
    counts.columns = ["ft>all (/5)", "all>non (/5)", "all>img (/5)", "all_three (/5)"]

    available_mm = [m for m in ["img", "no_text", "non", "text_only", "all", "ft"]
                    if m in avg[MM].unique()]
    means = avg.pivot_table(index=DATASET, columns=MM, values=TEST_SCORE, aggfunc="mean")[
        available_mm
    ].round(3)

    summary = counts.join(means)
    score_cols = [c for c in ["all", "ft"] if c in summary.columns]
    summary["_sort_key"] = summary[score_cols].max(axis=1)
    return summary.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])


def _display_viewer(df: pd.DataFrame, summary: pd.DataFrame):
    for dataset, row in summary.iterrows():
        st.markdown(f"### {dataset}")
        modality = infer_modality(dataset)
        ft_all_count = int(row['ft>all (/5)'])
        all_non_count = int(row['all>non (/5)'])
        all_uni_count = int(row['all>img (/5)'])
        all_three = int(row['all_three (/5)'])
        ft_all = badge(ft_all_count)
        all_non = badge(all_non_count)
        all_uni = badge(all_uni_count)
        if modality == MODALITY_TEXT:
            caption = f"Finetuned > All: {ft_all}  |  All > No Text: {all_non}  |  All > Text Only: {all_uni}  |  {badge(all_three)} all three: {all_three}/5"
        else:
            caption = f"Finetuned > All: {ft_all}  |  All > No Image: {all_non}  |  All > Image Only: {all_uni}  |  {badge(all_three)} all three: {all_three}/5"
        st.caption(caption)
        passes = all_three >= 3
        if dataset in _CORE_IMAGE_NAMES and not passes:
            st.error(f"INCONSISTENCY: {dataset} is in MULTABENCH_CORE_IMAGE but only {all_three}/5 models satisfy all three conditions")
        df_ds = df[df[DATASET] == dataset]
        modality = infer_modality(dataset)
        conditions = _TEXT_CONDITIONS if modality == MODALITY_TEXT else _IMAGE_CONDITIONS
        plot_df = df_ds.groupby([MODEL, MODE]).agg({TEST_SCORE: "mean"}).reset_index()
        plot_dataset_performance(plot_df, conditions=conditions, title="")
        fold_df = df_ds.pivot_table(index=FOLD, columns=[MODEL, MODE], values=TEST_SCORE).round(3)
        st.dataframe(fold_df, use_container_width=True)
        st.divider()
