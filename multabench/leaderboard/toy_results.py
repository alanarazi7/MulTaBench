import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

from multabench.leaderboard.data.keys import MM as MULTIMODAL_STATE, TEST_SCORE

TOY_RESULTS_PATH = Path(__file__).parent / "data" / "toy_results" / "toy_results.csv"

MODALITY = "modality"
TOY_TYPE = "toy_type"
FROZEN = "Frozen"
FINETUNED = "Finetuned"

IMAGE_TOYS = {"quadrant", "complexity", "mass_ratio"}
TEXT_TOYS = {"first_letter", "contains_token", "capitalized_count"}

TASK_LABELS = {
    "quadrant": "Quadrant",
    "complexity": "Density",
    "mass_ratio": "Aspect Ratio",
    "first_letter": "First Letter",
    "contains_token": "Token Presence",
    "capitalized_count": "Capitalized Count",
}

TOY_ORDER = ["quadrant", "complexity", "mass_ratio", "first_letter", "contains_token", "capitalized_count"]


def load_toy_data() -> pd.DataFrame:
    df = pd.read_csv(TOY_RESULTS_PATH)
    df[TOY_TYPE] = df["Name"].str.extract(r"toy_(\w+)_MUL")
    df[MODALITY] = df[TOY_TYPE].apply(lambda t: "Image" if t in IMAGE_TOYS else "Text")
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.groupby([TOY_TYPE, MULTIMODAL_STATE])[TEST_SCORE].mean().unstack()
    pivot.columns = [FROZEN if c == "all" else FINETUNED for c in pivot.columns]
    pivot[MODALITY] = pivot.index.map(lambda t: "Image" if t in IMAGE_TOYS else "Text")
    pivot.index = pivot.index.map(TASK_LABELS)
    pivot = pivot.loc[[TASK_LABELS[t] for t in TOY_ORDER]]
    return pivot.round(3)


def plot_toy_bars(summary: pd.DataFrame):
    fig = go.Figure()
    colors = {FROZEN: "#7eb0d5", FINETUNED: "#fd7f6f"}

    for col, color in colors.items():
        fig.add_trace(go.Bar(
            name=col,
            x=summary.index.tolist(),
            y=summary[col],
            marker_color=color,
            text=summary[col].round(3),
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.08]),
        xaxis=dict(tickangle=-20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(t=60, b=80),
    )
    return fig


def display_toy_results():
    st.title("🧸 Toy Tasks")
    df = load_toy_data()
    summary = build_summary(df)

    st.plotly_chart(plot_toy_bars(summary), use_container_width=True)

    st.divider()

    col_img, col_txt = st.columns(2)
    with col_img:
        st.markdown("#### Image tasks (DINO)")
        img_rows = [TASK_LABELS[t] for t in TOY_ORDER if t in IMAGE_TOYS]
        st.dataframe(summary.loc[img_rows][[FROZEN, FINETUNED]])
    with col_txt:
        st.markdown("#### Text tasks (E5)")
        txt_rows = [TASK_LABELS[t] for t in TOY_ORDER if t in TEXT_TOYS]
        st.dataframe(summary.loc[txt_rows][[FROZEN, FINETUNED]])

    st.divider()
    st.markdown("#### Per-fold breakdown")
    fold = st.selectbox("Fold", options=["All"] + list(range(5)), key="toy_fold")
    df_f = df if fold == "All" else df[df["fold"] == fold]
    detail = df_f.groupby([TOY_TYPE, MULTIMODAL_STATE])[TEST_SCORE].mean().unstack().round(3)
    detail.columns = [FROZEN if c == "all" else FINETUNED for c in detail.columns]
    st.dataframe(detail)
