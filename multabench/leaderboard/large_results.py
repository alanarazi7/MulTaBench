import pandas as pd
import streamlit as st

from multabench.leaderboard.data.keys import (
    MODE, MODEL, DATASET, FOLD, TEST_SCORE, MM, DINO_TUNE,
    ALL_FEAT, FINETUNED,
    E5_MODEL, E5_SMALL, E5_LARGE, E5_SMALL_ALL, E5_SMALL_FT, E5_LARGE_ALL, E5_LARGE_FT,
    DINO_MODEL, DINO_SMALL, DINO_LARGE,
    DINO_SMALL_ALL, DINO_SMALL_FT, DINO_LARGE_ALL, DINO_LARGE_FT,
    MODALITY, MODALITY_TEXT, MODALITY_IMAGE,
)
from multabench.leaderboard.data.loading import load_multimodal_leaderboard_data, load_large_results_data, load_multabench_data
from multabench.leaderboard.plots import plot_dataset_performance, plot_normalized_overview
from multabench.leaderboard.utils import infer_modality, badge

DINO_CONDITIONS = [DINO_SMALL_ALL, DINO_SMALL_FT, DINO_LARGE_ALL, DINO_LARGE_FT]
E5_CONDITIONS = [E5_SMALL_ALL, E5_SMALL_FT, E5_LARGE_ALL, E5_LARGE_FT]


def display_large_results():
    st.title("🦣 Large Models")

    large_df = load_large_results_data()
    text_df = load_multimodal_leaderboard_data()

    if large_df.empty:
        st.warning("No large results data found.")
        return

    large_df[MODALITY] = large_df[DATASET].apply(infer_modality)
    large_df[MODE] = large_df[MM].map({"all": ALL_FEAT, "ft": FINETUNED})

    modality = st.radio("Modality", ["All", MODALITY_IMAGE, MODALITY_TEXT], horizontal=True, key="large_modality")
    show_image = modality in {"All", MODALITY_IMAGE}
    show_text = modality in {"All", MODALITY_TEXT}

    multabench_df = load_multabench_data()
    dino_df, dino_large_df = _prepare_dino_data(large_df, multabench_df)
    e5_df, text_large_df = _prepare_e5_data(large_df, text_df)

    # ── Summary overview plots ────────────────────────────────────────────────
    if show_image and not dino_df.empty:
        n_dino = dino_df[DATASET].nunique()
        st.subheader(f"🖼 DINO ({n_dino} datasets)")
        plot_normalized_overview(dino_df.drop(columns=[MODE], errors="ignore").rename(columns={"dino_mode": MODE}),
                                 conditions=DINO_CONDITIONS, key="dino")

    if show_text and not e5_df.empty:
        n_e5 = e5_df[DATASET].nunique()
        st.subheader(f"💬 E5 Large ({n_e5} datasets)")
        plot_normalized_overview(e5_df.drop(columns=[MODE], errors="ignore").rename(columns={"e5_mode": MODE}),
                                 conditions=E5_CONDITIONS, key="e5")

    # ── Per-dataset breakdowns ────────────────────────────────────────────────
    if show_image:
        st.divider()
        st.subheader("🖼 Image — Per Dataset")
        if dino_df.empty:
            st.info("No DINO image data yet.")
        else:
            _render_image_datasets(dino_df, dino_large_df)

    if show_text:
        st.divider()
        st.subheader("💬 Text — Per Dataset")
        if e5_df.empty:
            st.info("No E5 large text data yet.")
        else:
            _render_text_datasets(e5_df, text_large_df)


def _prepare_dino_data(large_df: pd.DataFrame, multabench_df: pd.DataFrame):
    dino_large_df = large_df[large_df[MODALITY] == MODALITY_IMAGE].copy()
    dino_large_df = dino_large_df[dino_large_df[DINO_MODEL] == DINO_LARGE].copy()
    dino_large_df["dino_mode"] = dino_large_df[MM].map({"all": DINO_LARGE_ALL, "ft": DINO_LARGE_FT})

    large_datasets = set(dino_large_df[DATASET].unique())
    multabench_small_df = multabench_df[
        multabench_df[MM].isin(["all", "ft"]) &
        (multabench_df[DATASET].apply(infer_modality) == MODALITY_IMAGE) &
        multabench_df[DATASET].isin(large_datasets)
    ].copy()
    multabench_small_df[DINO_MODEL] = DINO_SMALL
    multabench_small_df["dino_mode"] = multabench_small_df.apply(_dino_mode_label, axis=1)

    df = pd.concat([multabench_small_df, dino_large_df], ignore_index=True)
    df = df[df["dino_mode"].notna()]
    return df, dino_large_df


def _prepare_e5_data(large_df: pd.DataFrame, text_df: pd.DataFrame):
    text_large_df = large_df[large_df[MODALITY] == MODALITY_TEXT].copy()
    large_datasets = set(text_large_df[DATASET].unique())
    baseline_df = text_df[text_df[MM].isin(["all", "ft"]) & text_df[DATASET].isin(large_datasets)].copy()
    df = pd.concat([baseline_df, text_large_df], ignore_index=True)
    df["e5_mode"] = df.apply(_e5_mode_label, axis=1)
    df = df[df["e5_mode"].notna()]
    return df, text_large_df


def _dino_badges(df_ds: pd.DataFrame) -> str:
    avg = (df_ds.groupby([MODEL, "dino_mode"])[TEST_SCORE]
               .mean()
               .reset_index()
               .pivot_table(index=MODEL, columns="dino_mode", values=TEST_SCORE))
    total = len(avg)

    def count_gt(col_a, col_b):
        if col_a not in avg.columns or col_b not in avg.columns:
            return 0
        mask = avg[col_a].notna() & avg[col_b].notna()
        return int((avg.loc[mask, col_a] > avg.loc[mask, col_b]).sum())

    return (f"Large ft > Large all: {badge(count_gt(DINO_LARGE_FT, DINO_LARGE_ALL), total)}  |  "
            f"Large all > Small all: {badge(count_gt(DINO_LARGE_ALL, DINO_SMALL_ALL), total)}  |  "
            f"Large ft > Small ft: {badge(count_gt(DINO_LARGE_FT, DINO_SMALL_FT), total)}")


def _e5_badges(df_ds: pd.DataFrame) -> str:
    avg = (df_ds.groupby([MODEL, "e5_mode"])[TEST_SCORE]
               .mean()
               .reset_index()
               .pivot_table(index=MODEL, columns="e5_mode", values=TEST_SCORE))
    total = len(avg)

    def count_gt(col_a, col_b):
        if col_a not in avg.columns or col_b not in avg.columns:
            return 0
        mask = avg[col_a].notna() & avg[col_b].notna()
        return int((avg.loc[mask, col_a] > avg.loc[mask, col_b]).sum())

    return (f"Large ft > Large all: {badge(count_gt(E5_LARGE_FT, E5_LARGE_ALL), total)}  |  "
            f"Large all > Small all: {badge(count_gt(E5_LARGE_ALL, E5_SMALL_ALL), total)}  |  "
            f"Large ft > Small ft: {badge(count_gt(E5_LARGE_FT, E5_SMALL_FT), total)}")


def _render_image_datasets(df: pd.DataFrame, dino_large_df: pd.DataFrame):
    for dataset in sorted(df[DATASET].unique()):
        st.markdown(f"#### {dataset}")
        large_models = dino_large_df[dino_large_df[DATASET] == dataset][MODEL].unique()
        df_ds = df[df[DATASET] == dataset]
        if len(large_models):
            df_ds = df_ds[df_ds[MODEL].isin(large_models)]
        st.caption(_dino_badges(df_ds))
        plot_df = df_ds.groupby([MODEL, "dino_mode"]).agg({TEST_SCORE: "mean"}).reset_index()
        plot_df = plot_df.rename(columns={"dino_mode": MODE})
        plot_dataset_performance(plot_df, conditions=DINO_CONDITIONS, title="")
        with st.expander("Show fold details"):
            fold_df = df_ds.pivot_table(index=FOLD, columns=[MODEL, "dino_mode"], values=TEST_SCORE).round(3)
            st.dataframe(fold_df, use_container_width=True)
        st.divider()


def _render_text_datasets(df: pd.DataFrame, text_large_df: pd.DataFrame):
    for dataset in sorted(df[DATASET].unique()):
        st.markdown(f"#### {dataset}")
        large_models = text_large_df[text_large_df[DATASET] == dataset][MODEL].unique()
        df_ds = df[df[DATASET] == dataset]
        if len(large_models):
            df_ds = df_ds[df_ds[MODEL].isin(large_models)]
        st.caption(_e5_badges(df_ds))
        plot_df = df_ds.groupby([MODEL, "e5_mode"]).agg({TEST_SCORE: "mean"}).reset_index()
        plot_df = plot_df.rename(columns={"e5_mode": MODE})
        plot_dataset_performance(plot_df, conditions=E5_CONDITIONS, title="")
        with st.expander("Show fold details"):
            fold_df = df_ds.pivot_table(index=FOLD, columns=[MODEL, "e5_mode"], values=TEST_SCORE).round(3)
            st.dataframe(fold_df, use_container_width=True)
        st.divider()



def _dino_mode_label(row) -> str | None:
    is_tuned = row.get(DINO_TUNE, False) in {"yes", "true", True}
    mm = str(row[MM]).lower()
    if is_tuned or mm == "ft":
        return DINO_SMALL_FT
    if mm == "all":
        return DINO_SMALL_ALL
    return None


def _e5_mode_label(row) -> str | None:
    mm = row[MM]
    e5 = row.get(E5_MODEL, E5_SMALL)
    if e5 == E5_LARGE and mm == "all":
        return E5_LARGE_ALL
    if e5 == E5_LARGE and mm == "ft":
        return E5_LARGE_FT
    if e5 == E5_SMALL and mm == "all":
        return E5_SMALL_ALL
    if e5 == E5_SMALL and mm == "ft":
        return E5_SMALL_FT
    return None
