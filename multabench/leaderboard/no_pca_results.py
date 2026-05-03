import streamlit as st

from multabench.leaderboard.data.keys import DATASET, FOLD, MODEL, MM, TEST_SCORE
from multabench.leaderboard.data.loading import load_no_pca_data
from multabench.leaderboard.plots import plot_normalized_overview, plot_dataset_performance

FT_MODE = "ft"
ALL_MODE = "all"
CONDITIONS = [ALL_MODE, FT_MODE]


def _model_table(df):
    tbl = (
        df.groupby([MODEL, "mode"])[TEST_SCORE]
        .mean()
        .unstack("mode")
        .reindex(columns=[c for c in CONDITIONS if c in df["mode"].unique()])
        .round(3)
    )
    tbl.index = tbl.index.map(lambda x: x.split()[0])
    st.dataframe(tbl, use_container_width=True)


def display_no_pca_results():
    st.title("🔓 No-PCA Results")
    raw = load_no_pca_data()
    if raw.empty:
        st.warning("No no-PCA data found.")
        return

    st.caption(
        f"PCA and StandardScaler disabled — raw 384-dim DINO/E5 embeddings sent directly to the model. "
        f"**{raw[DATASET].nunique()} datasets · {raw[MODEL].nunique()} models** · partial data (sweep in progress)."
    )

    models = sorted(raw[MODEL].apply(lambda x: x.split()[0]).unique())
    sel = st.multiselect("Filter models", models, default=models, key="no_pca_models")
    df = raw[raw[MODEL].apply(lambda x: x.split()[0]).isin(sel)].copy()
    df["mode"] = df[MM]

    if df.empty:
        st.info("No data for this selection.")
        return

    tabs = st.tabs(["Overview", "Per-dataset"])

    with tabs[0]:
        st.subheader("ft vs all — no PCA")
        plot_normalized_overview(df, CONDITIONS, key="no_pca_overview")
        st.divider()
        st.markdown("**Mean score per model (averaged across available datasets × folds)**")
        _model_table(df)

    with tabs[1]:
        st.subheader("Per-dataset breakdown")
        for dataset in sorted(df[DATASET].unique()):
            with st.expander(dataset, expanded=False):
                df_ds = df[df[DATASET] == dataset]
                plot_df = df_ds.groupby([MODEL, "mode"])[TEST_SCORE].mean().reset_index()
                plot_dataset_performance(plot_df, conditions=CONDITIONS, title="", bar_width=0.18)
                fold_df = df_ds.pivot_table(
                    index=FOLD, columns=[MODEL, "mode"], values=TEST_SCORE, aggfunc="mean"
                ).round(3)
                st.dataframe(fold_df, use_container_width=True)
