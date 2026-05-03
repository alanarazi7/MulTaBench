"""Paper figure: encoder scale robustness (§6.1).

Two stacked bars per model (small / large encoder), consistent with the
leaderboard figure style. Orange TAR bar drawn first (full 0→ft), then blue
frozen bar drawn on top (narrower, 0→all). Darker shades for large encoder.
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
_COLOR_TAR_SMALL    = "#E8722A"
_COLOR_FROZEN_LARGE = "#3A88C8"
_COLOR_TAR_LARGE    = "#B85010"

_FS_TITLE   = 13
_FS_XLABEL  = 14
_FS_YTICK   = 12
_FS_XTICK   = 11
_FS_LEGEND  = 13
_FONTWEIGHT = "bold"
_CAPSIZE    = 2
_ERR_LW     = 0.9

_BAR_H = 0.28
_DY    = 0.19

DINO_SMALL_ALL = "DINO-small (all)"
DINO_SMALL_FT  = "DINO-small (ft)"
DINO_LARGE_ALL = "DINO-large (all)"
DINO_LARGE_FT  = "DINO-large (ft)"
E5_SMALL_ALL   = "E5-small (all)"
E5_SMALL_FT    = "E5-small (ft)"
E5_LARGE_ALL   = "E5-large (all)"
E5_LARGE_FT    = "E5-large (ft)"

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


def _aggregate(df: pd.DataFrame, conditions: list) -> pd.DataFrame:
    df = df.copy()
    df["model_label"] = df["model"].map(_MODEL_LABELS).fillna(df["model"].str.split().str[0])
    agg = (df.groupby(["model_label", "mode"])["norm"]
             .agg(mean="mean", std="std", n="count")
             .reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5
    return agg[agg["mode"].isin(conditions)]


def _plot_panel(ax, agg: pd.DataFrame, small_all: str, small_ft: str,
                large_all: str, large_ft: str, title: str):
    models = sorted(agg["model_label"].unique())
    y = np.arange(len(models))

    def _get(mode):
        return (agg[agg["mode"] == mode]
                .set_index("model_label").reindex(models).reset_index())

    sa = _get(small_all)
    sf = _get(small_ft)
    la = _get(large_all)
    lf = _get(large_ft)

    for row_sa, row_sf, row_la, row_lf, yi in zip(
            sa.itertuples(), sf.itertuples(), la.itertuples(), lf.itertuples(), y):

        ypos_s = yi + _DY
        if pd.notna(row_sf.mean):
            ax.barh(ypos_s, row_sf.mean, _BAR_H, color=_COLOR_TAR_SMALL,
                    edgecolor="black", linewidth=0.6, zorder=2)
            ax.errorbar(row_sf.mean, ypos_s, xerr=row_sf.ci,
                        fmt="none", ecolor="#B85010", capsize=_CAPSIZE, linewidth=_ERR_LW, zorder=4)
        if pd.notna(row_sa.mean):
            ax.barh(ypos_s, row_sa.mean, _BAR_H, color=_COLOR_FROZEN_SMALL,
                    edgecolor="black", linewidth=0.6, zorder=3)
            ax.errorbar(row_sa.mean, ypos_s, xerr=row_sa.ci,
                        fmt="none", ecolor="#3A88C8", capsize=_CAPSIZE, linewidth=_ERR_LW, zorder=4)

        ypos_l = yi - _DY
        if pd.notna(row_lf.mean):
            ax.barh(ypos_l, row_lf.mean, _BAR_H, color=_COLOR_TAR_LARGE,
                    edgecolor="black", linewidth=0.6, hatch="//", zorder=2)
            ax.errorbar(row_lf.mean, ypos_l, xerr=row_lf.ci,
                        fmt="none", ecolor="#7A3008", capsize=_CAPSIZE, linewidth=_ERR_LW, zorder=4)
        if pd.notna(row_la.mean):
            ax.barh(ypos_l, row_la.mean, _BAR_H, color=_COLOR_FROZEN_LARGE,
                    edgecolor="black", linewidth=0.6, hatch="//", zorder=3)
            ax.errorbar(row_la.mean, ypos_l, xerr=row_la.ci,
                        fmt="none", ecolor="#1A5888", capsize=_CAPSIZE, linewidth=_ERR_LW, zorder=4)

    ax.set_title(title, fontsize=_FS_TITLE, fontweight=_FONTWEIGHT, pad=7)
    ax.set_xlabel("Normalized Score", fontsize=_FS_XLABEL, fontweight=_FONTWEIGHT)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=_FS_YTICK, fontweight=_FONTWEIGHT)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"],
                       fontsize=_FS_XTICK, fontweight=_FONTWEIGHT)
    ax.set_ylim(y[0] - _DY - _BAR_H * 0.6, y[-1] + _DY + _BAR_H * 0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def make_figure(task_type: str = "all"):
    dino_df = _build_dino_df(task_type)
    e5_df   = _build_e5_df(task_type)

    dino_norm = _normalize_within_model(dino_df)
    e5_norm   = _normalize_within_model(e5_df)

    dino_agg = _aggregate(dino_norm, [DINO_SMALL_ALL, DINO_SMALL_FT, DINO_LARGE_ALL, DINO_LARGE_FT])
    e5_agg   = _aggregate(e5_norm,   [E5_SMALL_ALL,   E5_SMALL_FT,   E5_LARGE_ALL,   E5_LARGE_FT])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.subplots_adjust(wspace=0.45, left=0.10, right=0.82, top=0.90, bottom=0.12)

    _plot_panel(ax1, dino_agg, DINO_SMALL_ALL, DINO_SMALL_FT, DINO_LARGE_ALL, DINO_LARGE_FT,
                "(a) Image-Tabular (DINO-v3)")
    _plot_panel(ax2, e5_agg,   E5_SMALL_ALL,   E5_SMALL_FT,   E5_LARGE_ALL,   E5_LARGE_FT,
                "(b) Text-Tabular (E5-v2)")

    legend_handles = [
        Patch(facecolor=_COLOR_FROZEN_SMALL, edgecolor="black", linewidth=0.6, label="Frozen Small"),
        Patch(facecolor=_COLOR_TAR_SMALL,    edgecolor="black", linewidth=0.6, label="TAR Small"),
        Patch(facecolor=_COLOR_FROZEN_LARGE, edgecolor="black", linewidth=0.6, hatch="//", label="Frozen Large"),
        Patch(facecolor=_COLOR_TAR_LARGE,    edgecolor="black", linewidth=0.6, hatch="//", label="TAR Large"),
    ]
    fig.legend(handles=legend_handles,
               loc="center left", bbox_to_anchor=(0.83, 0.5),
               frameon=True, edgecolor="black", framealpha=0.95,
               prop={"weight": "bold", "size": _FS_LEGEND})

    return fig, {"DINO": dino_agg, "E5": e5_agg}
