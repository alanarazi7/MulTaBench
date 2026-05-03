"""Paper figure: curation protocol example — rejected vs accepted dataset (§3.2)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

COLORS = ['#FFCC99', '#FFD1DC', '#8DE5A1', '#A1C9F4']

_CONDITIONS = ["Unimodal Structured", "Unimodal Unstructured", "Joint Frozen", "Joint TAR"]

_FS_TITLE    = 13
_FS_YLABEL   = 12
_FS_XTICK    = 11
_FS_YTICK    = 11
_FS_BARLABEL = 10
_FS_LEGEND   = 11
_FONTWEIGHT  = "bold"

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
    return df.dropna(subset=["condition", "model_label"])


def _load_chexpert() -> pd.DataFrame:
    path = os.path.join(_RESULTS, "images", "MUL_IMAGE_CHEXPERT.csv")
    df = pd.read_csv(path)
    df["condition"] = df["multimodal_state"].map(_IMAGE_COND_MAP)
    df["model_label"] = df["model"].map(_MODEL_LABELS)
    return df.dropna(subset=["condition", "model_label"])


def _annotate_jc_gap(ax, agg, x, bar_width, offset):
    """Draw red ✗ between Joint and TAR bars where Joint > TAR."""
    joint_i = _CONDITIONS.index("Joint Frozen")
    ctx_i   = _CONDITIONS.index("Joint TAR")
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
        x_j   = x[xi] + (joint_i - offset) * bar_width + bar_width / 2
        x_c   = x[xi] + (ctx_i   - offset) * bar_width + bar_width / 2
        x_mid = (x_j + x_c) / 2
        ax.annotate(
            "",
            xy=(x_mid, c + 0.003),
            xytext=(x_mid, j - 0.003),
            arrowprops=dict(arrowstyle="|-|", color=_GAP_COLOR, lw=1.5, mutation_scale=4),
        )
        ax.text(x_mid + 0.06, (j + c) / 2, "✗",
                ha="left", va="center", fontsize=18, color=_GAP_COLOR, fontweight=_FONTWEIGHT)
        annotated.append((xi, model))
    return annotated


def _annotate_ctx_wins(ax, agg, x, bar_width, offset):
    """Draw green ✓ to the left of the TAR bar for all models."""
    joint_vals = (agg[agg["condition"] == "Joint Frozen"]
                  .set_index("model_label").reindex(_MODEL_ORDER)["mean"].fillna(0))
    ctx_vals = (agg[agg["condition"] == "Joint TAR"]
                .set_index("model_label").reindex(_MODEL_ORDER)["mean"].fillna(0))

    for xi, model in enumerate(_MODEL_ORDER):
        j = joint_vals[model]
        c = ctx_vals[model]
        ctx_left = x[xi] + (3 - offset) * bar_width
        x_arrow  = ctx_left - 0.20
        ax.annotate(
            "",
            xy=(x_arrow, j + 0.003),
            xytext=(x_arrow, c - 0.003),
            arrowprops=dict(arrowstyle="|-|", color=_CHECK_COLOR, lw=1.5, mutation_scale=4),
        )
        ax.text(x_arrow - 0.01, (j + c) / 2, "✓",
                ha="right", va="center", fontsize=18, color=_CHECK_COLOR, fontweight=_FONTWEIGHT)


def _panel(ax, df: pd.DataFrame, title: str, annotate_jc_gap: bool = False,
           mirror_models=None, yticks=None, ylim=None):
    agg = (df.groupby(["model_label", "condition"])["test_score"]
             .mean().reset_index().rename(columns={"test_score": "mean"}))
    floor   = agg["mean"].min()
    ceiling = agg["mean"].max()

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
        annotated = _annotate_jc_gap(ax, agg, x, bar_width, offset)
    elif mirror_models is not None:
        _annotate_ctx_wins(ax, agg, x, bar_width, offset)

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

    fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(12, 3.6))

    annotated = _panel(ax_bad, osha, "✗ OSHA Accident Injury",
                       annotate_jc_gap=True,
                       yticks=[0.56, 0.58, 0.60], ylim=[0.556, 0.612])
    _panel(ax_good, chexpert, "✓ CheXpert",
           mirror_models=annotated,
           yticks=[0.70, 0.75, 0.80], ylim=[0.690, 0.825])

    handles = [Patch(color=COLORS[i], label=c) for i, c in enumerate(_CONDITIONS)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=True, edgecolor="black", framealpha=0.95,
               prop={"weight": "bold", "size": _FS_LEGEND})

    plt.tight_layout()
    return fig
