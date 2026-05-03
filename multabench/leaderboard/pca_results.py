import streamlit as st

from multabench.leaderboard.data.keys import DATASET, FOLD, MODEL, MM, TEST_SCORE, MODE, PCA_COMPONENTS
from multabench.leaderboard.data.loading import load_pca_comparison_data
from multabench.leaderboard.plots import plot_normalized_overview, plot_dataset_performance

FT_MODE  = "ft"
ALL_MODE = "all"
CONDITIONS = [ALL_MODE, FT_MODE]


def _model_table(df):
    """Mean test score per model × mode, averaged across datasets and folds."""
    tbl = (
        df.groupby([MODEL, MODE])[TEST_SCORE]
        .mean()
        .unstack(MODE)
        .reindex(columns=[c for c in CONDITIONS if c in df[MODE].unique()])
        .round(3)
    )
    tbl.index = tbl.index.map(lambda x: x.split()[0])
    st.dataframe(tbl, use_container_width=True)


def display_pca_comparison():
    st.title("📐 PCA Dimensions Comparison")
    raw = load_pca_comparison_data()
    if raw.empty:
        st.warning("No PCA comparison data found.")
        return

    st.caption(
        f"Compares **ft** vs **all** for each PCA dimension "
        f"across {raw[DATASET].nunique()} datasets and {raw[MODEL].nunique()} models. "
        "pca=30 is the curation baseline."
    )

    # Controls row
    c1, c2 = st.columns([1, 3])
    with c1:
        available_dims = sorted(raw[PCA_COMPONENTS].astype(int).unique())
        dim = st.radio("PCA dimensions", available_dims, index=1, key="pca_dim",
                       format_func=lambda d: f"{d} (baseline)" if d == 30 else str(d))
    with c2:
        models = sorted(raw[MODEL].apply(lambda x: x.split()[0]).unique())
        sel = st.multiselect("Filter models", models, default=models, key="pca_models")

    df = raw[raw[PCA_COMPONENTS].astype(int) == dim].copy()
    df[MODE] = df[MM]
    df = df[df[MODEL].apply(lambda x: x.split()[0]).isin(sel)]

    if df.empty:
        st.info("No data for this selection.")
        return

    tabs = st.tabs(["Overview", "Per-dataset"])

    with tabs[0]:
        st.subheader(f"ft vs all — pca={dim}")
        plot_normalized_overview(df, CONDITIONS, key=f"pca_overview_{dim}")
        st.divider()
        st.markdown("**Mean score per model (averaged across datasets × folds)**")
        _model_table(df)

    with tabs[1]:
        st.subheader(f"Per-dataset breakdown — pca={dim}")
        for dataset in sorted(df[DATASET].unique()):
            with st.expander(dataset, expanded=False):
                df_ds = df[df[DATASET] == dataset]
                plot_df = df_ds.groupby([MODEL, MODE])[TEST_SCORE].mean().reset_index()
                plot_dataset_performance(plot_df, conditions=CONDITIONS, title="", bar_width=0.18)
                fold_df = df_ds.pivot_table(
                    index=FOLD, columns=[MODEL, MODE], values=TEST_SCORE, aggfunc="mean"
                ).round(3)
                st.dataframe(fold_df, use_container_width=True)
