"""Paper figure: curation protocol example — rejected vs accepted dataset (§3.2)."""
import os
import numpy as np
import pandas as pd
import matplotlib
import streamlit
import streamlit as st

from multabench.leaderboard.data.keys import TEST_SCORE, FOLD

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

COLORS = ['#FFCC99', '#FFD1DC', '#8DE5A1', '#A1C9F4']

_CONDITIONS = ["Unimodal Structured", "Unimodal Unstructured", "Joint Frozen", "Joint TAR"]

_FS_TITLE    = 15
_FS_YLABEL   = 14
_FS_XTICK    = 13
_FS_YTICK    = 13
_FS_BARLABEL = 10
_FS_LEGEND   = 13
_FONTWEIGHT  = "normal"

_MODEL_LABELS = {
    "LightGBM 💡":     "LightGBM",
    "CatBoost 😸":     "CatBoost",
    "TabM Ⓜ️":        "TabM",
    "TabPFN-v2 🤯":    "TabPFNv2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
}
_MODEL_ORDER = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]

_TEXT_COND_MAP = {
    "no_text":   "Unimodal Structured",
    "text_only": "Unimodal Unstructured",
    "all":       "Joint Frozen",
    "ft":        "Joint TAR",
}
_IMAGE_COND_MAP = {
    "non": "Unimodal Structured",
    "img": "Unimodal Unstructured",
    "all": "Joint Frozen",
    "ft":  "Joint TAR",
}

_GAP_COLOR   = "#c0392b"
_CHECK_COLOR = "#27ae60"


def _load_osha() -> pd.DataFrame:
    path = os.path.join(_RESULTS, "tabstar_corpus", "texttabench_datasets.csv")
    df = pd.read_csv(path)
    df = df[df["dataset"] == "BIN_TEXT_TRANSPORTATION_OSHA_ACCIDENT_INJURY_DATA"].copy()
    df["condition"] = df["multimodal_state"].map(_TEXT_COND_MAP)
    df["model_label"] = df["model"].map(_MODEL_LABELS)
    assert len(df) == 100
    df = df[['condition', 'model_label', TEST_SCORE, FOLD]]
    return df


def _load_chexpert() -> pd.DataFrame:
    path = os.path.join(_RESULTS, "images", "MUL_IMAGE_CHEXPERT.csv")
    df = pd.read_csv(path)
    df["condition"] = df["multimodal_state"].map(_IMAGE_COND_MAP)
    df["model_label"] = df["model"].map(_MODEL_LABELS)
    assert len(df) == 100
    df = df[['condition', 'model_label', TEST_SCORE, FOLD]]
    return df


def _annotate_jc_gap(ax, agg, x, bar_width, offset, margin):
    """Draw red ✗ above the Joint TAR bar (the shorter one) where Joint Frozen > Joint TAR."""
    ctx_i = _CONDITIONS.index("Joint TAR")
    joint_vals = (agg[agg["condition"] == "Joint Frozen"]
                  .set_index("model_label").reindex(_MODEL_ORDER)["mean"].fillna(0))
    ctx_vals = (agg[agg["condition"] == "Joint TAR"]
                .set_index("model_label").reindex(_MODEL_ORDER)["mean"].fillna(0))

    annotated = []
    for xi, model in enumerate(_MODEL_ORDER):
        j = joint_vals[model]
        c = ctx_vals[model]
        if j - c < 0.001:
            continue
        x_c = x[xi] + (ctx_i - offset) * bar_width + bar_width / 2
        ax.text(x_c, c + margin, "✗",
                ha="center", va="bottom", fontsize=18, color=_GAP_COLOR)
        annotated.append((xi, model))
    return annotated


def _annotate_ctx_wins(ax, agg, x, bar_width, offset, margin):
    """Draw green ✓ above the Joint Frozen bar (the shorter one) for all models."""
    joint_i = _CONDITIONS.index("Joint Frozen")
    joint_vals = (agg[agg["condition"] == "Joint Frozen"]
                  .set_index("model_label").reindex(_MODEL_ORDER)["mean"].fillna(0))

    for xi, model in enumerate(_MODEL_ORDER):
        j = joint_vals[model]
        x_j = x[xi] + (joint_i - offset) * bar_width  # left edge of Joint Frozen bar
        ax.text(x_j, j + margin, "✓",
                ha="center", va="bottom", fontsize=18, color=_CHECK_COLOR)


def _panel(ax, df: pd.DataFrame, title: str, annotate_jc_gap: bool = False,
           mirror_models=None, yticks=None, ylim=None):
    agg = (df.groupby(["model_label", "condition"])[TEST_SCORE]
             .mean().reset_index().rename(columns={TEST_SCORE: "mean"}))
    floor   = agg["mean"].min()
    ceiling = agg["mean"].max()
    y_range = (ylim[1] - ylim[0]) if ylim else (ceiling - floor + 0.03)
    margin  = y_range * 0.04

    x = np.arange(len(_MODEL_ORDER))
    bar_width = 0.18
    offset = (len(_CONDITIONS) - 1) / 2

    for i, cond in enumerate(_CONDITIONS):
        subset = (agg[agg["condition"] == cond]
                  .set_index("model_label").reindex(_MODEL_ORDER).reset_index())
        ax.bar(x + (i - offset) * bar_width, subset["mean"], bar_width,
               label=cond, color=COLORS[i], edgecolor="white", linewidth=1.2)

    annotated = []
    if annotate_jc_gap:
        annotated = _annotate_jc_gap(ax, agg, x, bar_width, offset, margin)
    elif mirror_models is not None:
        _annotate_ctx_wins(ax, agg, x, bar_width, offset, margin)

    color = "#c0392b" if title.startswith("✗") else "#27ae60"
    ax.set_title(title, fontsize=_FS_TITLE, fontweight=_FONTWEIGHT, pad=8, color=color)
    ax.set_ylabel("AUC", fontsize=_FS_YLABEL, fontweight=_FONTWEIGHT)
    ax.set_xticks(x)
    ax.set_xticklabels(_MODEL_ORDER, rotation=0, ha="center",
                       fontsize=_FS_XTICK, fontweight=_FONTWEIGHT)
    ax.tick_params(axis="y", labelsize=_FS_YTICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(floor - 0.01, ceiling + 0.02)
    if yticks is not None:
        ax.set_yticks(yticks)
    return annotated


def make_fig() -> plt.Figure:
    osha     = _load_osha()
    chexpert = _load_chexpert()

    fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 3.6))

    annotated = _panel(ax_bad, osha, "✗ OSHA Accident Injury",
                       annotate_jc_gap=True,
                       yticks=[0.56, 0.58, 0.60], ylim=[0.556, 0.612])
    _panel(ax_good, chexpert, "✓ CheXpert",
           mirror_models=annotated,
           yticks=[0.70, 0.75, 0.80], ylim=[0.690, 0.825])

    handles = [Patch(color=COLORS[i], label=c) for i, c in enumerate(_CONDITIONS)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=True, edgecolor="black", framealpha=0.95,
               prop={"size": _FS_LEGEND})

    plt.tight_layout()

    with streamlit.expander("Data"):
        st.markdown("### Osha")
        st.table(osha.pivot_table(index="model_label", columns="condition", values=TEST_SCORE))
        st.markdown("### CheXpert")
        st.table(chexpert.pivot_table(index="model_label", columns="condition", values=TEST_SCORE))
    return fig
