"""Paper figure: PCA dimensionality robustness (§6.1).

Six conditions: 15/30/60 PCA dims × Frozen/TAR, all 40 datasets
(image + text) combined. Scores are normalized within each (dataset, fold,
model) group across all 6 conditions. Grouped vertical bar chart, one group
per model.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
_PCA_DIR = os.path.join(_RESULTS, "analysis_pca")

# Blue family = Frozen, orange family = TAR
# Shade: light=15 dims, medium=30 dims, dark=60 dims
_COLORS = {
    "Frozen N=15": "#C8E4F8",
    "Frozen N=30": "#A8D4F0",
    "Frozen N=60": "#6AAED6",
    "TAR N=15":    "#FBCFAA",
    "TAR N=30":    "#E8722A",
    "TAR N=60":    "#C05010",
}
_CI_COLORS = {
    "Frozen N=15": "#5A9EC8",
    "Frozen N=30": "#3A88C8",
    "Frozen N=60": "#1A5888",
    "TAR N=15":    "#D07030",
    "TAR N=30":    "#B85010",
    "TAR N=60":    "#7A3008",
}

_FS_TITLE   = 15
_FS_YLABEL  = 15
_FS_XTICK   = 13
_FS_YTICK   = 13
_FS_LEGEND  = 13
_FONTWEIGHT = "normal"
_CAPSIZE    = 2
_ERR_LW     = 0.7

CONDITIONS = list(_COLORS.keys())

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM", "TabPFN-v2 🤯": "TabPFNv2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
}


def _load_pca_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "dataset_name" in df.columns:
        df = df.rename(columns={"dataset_name": "dataset"})
    df["model"]      = df["model"].str.strip()
    df["test_score"] = df["test_score"].clip(lower=-0.1)
    return df[["dataset", "fold", "model", "test_score"]]


def _load_dir_state(path: str, state: str) -> pd.DataFrame:
    frames = []
    for f in os.listdir(path):
        if not f.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(path, f))
        if "dataset" not in df.columns:
            df["dataset"] = f.replace(".csv", "")
        df["model"]      = df["model"].str.strip()
        df["test_score"] = df["test_score"].clip(lower=-0.1)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df[df["multimodal_state"] == state][["dataset", "fold", "model", "test_score"]]


def _build_df() -> pd.DataFrame:
    d30_all = pd.concat([
        _load_dir_state(os.path.join(_RESULTS, "images"), "all"),
        _load_dir_state(os.path.join(_RESULTS, "text"),   "all"),
    ], ignore_index=True)
    d30_ft = pd.concat([
        _load_dir_state(os.path.join(_RESULTS, "images"), "ft"),
        _load_dir_state(os.path.join(_RESULTS, "text"),   "ft"),
    ], ignore_index=True)

    pieces = {
        "Frozen N=15": _load_pca_csv(os.path.join(_PCA_DIR, "all_15.csv")),
        "Frozen N=30": d30_all,
        "Frozen N=60": _load_pca_csv(os.path.join(_PCA_DIR, "all_60.csv")),
        "TAR N=15":    _load_pca_csv(os.path.join(_PCA_DIR, "ft_15.csv")),
        "TAR N=30":    d30_ft,
        "TAR N=60":    _load_pca_csv(os.path.join(_PCA_DIR, "ft_60.csv")),
    }

    frames = []
    for cond, df in pieces.items():
        df = df.copy()
        df["condition"] = cond
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    counts = combined.groupby("dataset")["condition"].nunique()
    valid_ds = counts[counts == 6].index
    return combined[combined["dataset"].isin(valid_ds)]


def _normalize_and_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_label"] = df["model"].map(_MODEL_LABELS).fillna(df["model"].str.split().str[0])
    lo = df.groupby(["dataset", "fold", "model_label"])["test_score"].transform("min")
    hi = df.groupby(["dataset", "fold", "model_label"])["test_score"].transform("max")
    df["norm"] = (df["test_score"] - lo) / (hi - lo).clip(lower=1e-9)

    agg = (df.groupby(["model_label", "condition"])["norm"]
             .agg(mean="mean", std="std", n="count")
             .reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5
    return agg


def make_figure():
    df  = _build_df()
    agg = _normalize_and_agg(df)

    models = ["CatBoost", "LightGBM", "TabM", "TabPFNv2", "TabPFN-2.5"]
    n_conds  = len(CONDITIONS)

    width     = 0.12
    group_gap = 0.2
    group_w   = n_conds * width
    x_centers = np.arange(len(models)) * (group_w + group_gap)
    offsets   = (np.arange(n_conds) - (n_conds - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.subplots_adjust(left=0.07, right=0.78, top=0.90, bottom=0.14)

    for ci, cond in enumerate(CONDITIONS):
        color = _COLORS[cond]
        for mi, model in enumerate(models):
            row = agg[(agg["model_label"] == model) & (agg["condition"] == cond)]
            if row.empty:
                continue
            x = x_centers[mi] + offsets[ci]
            ax.bar(x, row["mean"].iloc[0], width,
                   color=color, edgecolor="black", linewidth=0.6, zorder=2)
            ax.errorbar(x, row["mean"].iloc[0], yerr=row["ci"].iloc[0],
                        fmt="none", ecolor=_CI_COLORS[cond], capsize=_CAPSIZE,
                        linewidth=_ERR_LW, zorder=3)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(models, fontsize=_FS_XTICK, fontweight=_FONTWEIGHT)
    ax.set_ylabel("Normalized Score", fontsize=_FS_YLABEL, fontweight=_FONTWEIGHT)
    ax.set_ylim(0, 1.01)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"],
                       fontsize=_FS_YTICK, fontweight=_FONTWEIGHT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    legend_handles = [Patch(facecolor=_COLORS[c], edgecolor="white", label=c) for c in CONDITIONS]
    fig.legend(handles=legend_handles,
               loc="center left", bbox_to_anchor=(0.79, 0.5),
               frameon=True, edgecolor="black", framealpha=0.95,
               prop={"size": _FS_LEGEND})

    return fig, agg
