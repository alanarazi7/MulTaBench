"""Paper figure: encoder scale robustness (§6.1).

Main figure: single combined panel (image + text), 4 non-stacked grouped vertical bars per
model (LightGBM → TabPFN-2.5, left to right), order Frozen Small, Frozen Large, TAR Small,
TAR Large within each group.
Appendix figures: separate DINO (image) and E5 (text) panels, horizontal bar style.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

_COLOR_FROZEN_SMALL = "#A8D4F0"
_COLOR_FROZEN_LARGE = "#3A88C8"
_COLOR_TAR_SMALL    = "#E8722A"
_COLOR_TAR_LARGE    = "#B85010"

_FS_TITLE   = 15
_FS_XLABEL  = 15
_FS_YTICK   = 13
_FS_XTICK   = 13
_FS_LEGEND  = 14
_FONTWEIGHT = "normal"
_CAPSIZE    = 2
_ERR_LW     = 0.9

_BAR_H     = 0.14
_BAR_GAP   = 0.03
_GROUP_GAP = 0.38

_GROUPED_CONDITIONS = ["Frozen Small", "Frozen Large", "TAR Small", "TAR Large"]
_GROUPED_COLORS     = [_COLOR_FROZEN_SMALL, _COLOR_FROZEN_LARGE, _COLOR_TAR_SMALL, _COLOR_TAR_LARGE]
_GROUPED_ERR_COLORS = ["#3A88C8", "#1A5888", "#B85010", "#7A3008"]

_MODELS          = ["TabPFN-2.5", "TabPFNv2", "TabM", "CatBoost", "LightGBM"]  # horizontal appendix (top→bottom)
_MODELS_VERTICAL = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]  # vertical main (left→right)

DINO_SMALL_ALL = "DINO-small (all)"
DINO_SMALL_FT  = "DINO-small (ft)"
DINO_LARGE_ALL = "DINO-large (all)"
DINO_LARGE_FT  = "DINO-large (ft)"
E5_SMALL_ALL   = "E5-small (all)"
E5_SMALL_FT    = "E5-small (ft)"
E5_LARGE_ALL   = "E5-large (all)"
E5_LARGE_FT    = "E5-large (ft)"

_MODE_TO_CONDITION = {
    DINO_SMALL_ALL: "Frozen Small", DINO_SMALL_FT: "TAR Small",
    DINO_LARGE_ALL: "Frozen Large", DINO_LARGE_FT: "TAR Large",
    E5_SMALL_ALL:   "Frozen Small", E5_SMALL_FT:   "TAR Small",
    E5_LARGE_ALL:   "Frozen Large", E5_LARGE_FT:   "TAR Large",
}

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM", "TabPFN-v2 🤯": "TabPFNv2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
}


def _load_dir(path: str, mm_filter=None) -> pd.DataFrame:
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
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if mm_filter:
        df = df[df["multimodal_state"].isin(mm_filter)]
    return df


def _task_filter(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    if task_type == "cls":
        return df[df["dataset"].str.startswith(("BIN_", "MUL_"))]
    if task_type == "reg":
        return df[df["dataset"].str.startswith("REG_")]
    return df


def _build_dino_df(task_type: str = "all") -> pd.DataFrame:
    large = _task_filter(_load_dir(os.path.join(_RESULTS, "images_large"), mm_filter=["all", "ft"]), task_type)
    small = _task_filter(_load_dir(os.path.join(_RESULTS, "images"),       mm_filter=["all", "ft"]), task_type)
    large_ds = set(large["dataset"].unique())
    small = small[small["dataset"].isin(large_ds)]
    small["mode"] = small["multimodal_state"].map({"all": DINO_SMALL_ALL, "ft": DINO_SMALL_FT})
    large["mode"] = large["multimodal_state"].map({"all": DINO_LARGE_ALL, "ft": DINO_LARGE_FT})
    return pd.concat([small[["dataset", "fold", "model", "mode", "test_score"]],
                      large[["dataset", "fold", "model", "mode", "test_score"]]],
                     ignore_index=True)


def _build_e5_df(task_type: str = "all") -> pd.DataFrame:
    large = _task_filter(_load_dir(os.path.join(_RESULTS, "text_large"), mm_filter=["all", "ft"]), task_type)
    small = _task_filter(_load_dir(os.path.join(_RESULTS, "text"),       mm_filter=["all", "ft"]), task_type)
    large_ds = set(large["dataset"].unique())
    small = small[small["dataset"].isin(large_ds)]
    small["mode"] = small["multimodal_state"].map({"all": E5_SMALL_ALL, "ft": E5_SMALL_FT})
    large["mode"] = large["multimodal_state"].map({"all": E5_LARGE_ALL, "ft": E5_LARGE_FT})
    return pd.concat([small[["dataset", "fold", "model", "mode", "test_score"]],
                      large[["dataset", "fold", "model", "mode", "test_score"]]],
                     ignore_index=True)


def _normalize_within_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lo = df.groupby(["dataset", "fold", "model"])["test_score"].transform("min")
    hi = df.groupby(["dataset", "fold", "model"])["test_score"].transform("max")
    df["norm"] = (df["test_score"] - lo) / (hi - lo).clip(lower=1e-9)
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_label"] = df["model"].map(_MODEL_LABELS).fillna(df["model"].str.split().str[0])
    df["condition"]   = df["mode"].map(_MODE_TO_CONDITION)
    agg = (df.groupby(["model_label", "condition"])["norm"]
             .agg(mean="mean", std="std", n="count")
             .reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5
    return agg[agg["condition"].isin(_GROUPED_CONDITIONS)]


def _plot_grouped_panel(ax, agg, title):
    n_conds  = len(_GROUPED_CONDITIONS)
    bar_slot = _BAR_H + _BAR_GAP
    group_h  = n_conds * bar_slot
    y_centers = np.arange(len(_MODELS)) * (group_h + _GROUP_GAP)

    for ci, (cond, color, ec) in enumerate(
            zip(_GROUPED_CONDITIONS, _GROUPED_COLORS, _GROUPED_ERR_COLORS)):
        y_offset = ((n_conds - 1) / 2 - ci) * bar_slot
        for mi, model in enumerate(_MODELS):
            row = agg[(agg["model_label"] == model) & (agg["condition"] == cond)]
            if row.empty:
                continue
            y = y_centers[mi] + y_offset
            ax.barh(y, row["mean"].iloc[0], _BAR_H, color=color,
                    edgecolor="black", linewidth=0.6, zorder=2)
            ax.errorbar(row["mean"].iloc[0], y, xerr=row["ci"].iloc[0],
                        fmt="none", ecolor=ec, capsize=_CAPSIZE, linewidth=_ERR_LW, zorder=3)

    if title:
        ax.set_title(title, fontsize=_FS_TITLE, fontweight=_FONTWEIGHT, pad=7)
    ax.set_xlabel("Normalized Score", fontsize=_FS_XLABEL, fontweight=_FONTWEIGHT)
    ax.set_yticks(y_centers)
    ax.set_yticklabels(_MODELS, fontsize=_FS_YTICK, fontweight=_FONTWEIGHT)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"],
                       fontsize=_FS_XTICK, fontweight=_FONTWEIGHT)
    half = group_h / 2
    ax.set_ylim(y_centers[0] - half - 0.08, y_centers[-1] + half + 0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def _plot_vertical_panel(ax, agg):
    n_conds  = len(_GROUPED_CONDITIONS)
    bar_slot = _BAR_H  # no gap — bars touch within each model group
    group_w  = n_conds * bar_slot
    x_centers = np.arange(len(_MODELS_VERTICAL)) * (group_w + _GROUP_GAP)

    for ci, (cond, color, ec) in enumerate(
            zip(_GROUPED_CONDITIONS, _GROUPED_COLORS, _GROUPED_ERR_COLORS)):
        x_offset = (ci - (n_conds - 1) / 2) * bar_slot  # ci=0 leftmost, ci=3 rightmost
        for mi, model in enumerate(_MODELS_VERTICAL):
            row = agg[(agg["model_label"] == model) & (agg["condition"] == cond)]
            if row.empty:
                continue
            x = x_centers[mi] + x_offset
            ax.bar(x, row["mean"].iloc[0], _BAR_H, color=color,
                   edgecolor="black", linewidth=0.6, zorder=2)
            ax.errorbar(x, row["mean"].iloc[0], yerr=row["ci"].iloc[0],
                        fmt="none", ecolor=ec, capsize=_CAPSIZE, linewidth=_ERR_LW, zorder=3)

    ax.set_ylabel("Normalized Score", fontsize=_FS_XLABEL, fontweight=_FONTWEIGHT)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(_MODELS_VERTICAL, fontsize=_FS_XTICK, fontweight=_FONTWEIGHT)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"],
                       fontsize=_FS_YTICK, fontweight=_FONTWEIGHT)
    half = group_w / 2
    ax.set_xlim(x_centers[0] - half - 0.1, x_centers[-1] + half + 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def _legend_handles():
    return [
        Patch(facecolor=_COLOR_FROZEN_SMALL, edgecolor="black", linewidth=0.6, label="Frozen Small"),
        Patch(facecolor=_COLOR_FROZEN_LARGE, edgecolor="black", linewidth=0.6, label="Frozen Large"),
        Patch(facecolor=_COLOR_TAR_SMALL,    edgecolor="black", linewidth=0.6, label="TAR Small"),
        Patch(facecolor=_COLOR_TAR_LARGE,    edgecolor="black", linewidth=0.6, label="TAR Large"),
    ]


def make_figure(task_type: str = "all"):
    """Main paper figure — combined image+text, single vertical-bar panel."""
    dino_norm = _normalize_within_model(_build_dino_df(task_type))
    e5_norm   = _normalize_within_model(_build_e5_df(task_type))
    combined  = pd.concat([dino_norm, e5_norm], ignore_index=True)
    agg = _aggregate(combined)

    fig, ax = plt.subplots(1, 1, figsize=(11, 3.8))
    fig.subplots_adjust(left=0.07, right=0.75, top=0.95, bottom=0.13)

    _plot_vertical_panel(ax, agg)

    fig.legend(handles=_legend_handles(),
               loc="center left", bbox_to_anchor=(0.76, 0.5),
               frameon=True, edgecolor="black", framealpha=0.95,
               prop={"size": _FS_LEGEND})

    return fig, {"Combined": agg}


def make_appendix_figure(task_type: str = "all"):
    """Appendix figure — separate DINO (image) and E5 (text) panels."""
    dino_df   = _build_dino_df(task_type)
    e5_df     = _build_e5_df(task_type)
    dino_norm = _normalize_within_model(dino_df)
    e5_norm   = _normalize_within_model(e5_df)
    dino_agg  = _aggregate(dino_norm)
    e5_agg    = _aggregate(e5_norm)

    n_img = dino_df["dataset"].nunique()
    n_txt = e5_df["dataset"].nunique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.subplots_adjust(wspace=0.42, left=0.10, right=0.82, top=0.90, bottom=0.12)

    _plot_grouped_panel(ax1, dino_agg, "(a) Image-Tabular (DINO-v3)")
    _plot_grouped_panel(ax2, e5_agg,   "(b) Text-Tabular (E5-v2)")

    fig.legend(handles=_legend_handles(),
               loc="center left", bbox_to_anchor=(0.83, 0.5),
               frameon=True, edgecolor="black", framealpha=0.95,
               prop={"size": _FS_LEGEND})

    return fig, {"DINO": dino_agg, "E5": e5_agg}
