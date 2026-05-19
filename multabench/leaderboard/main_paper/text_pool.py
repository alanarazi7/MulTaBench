"""Paper figures: text-tabular pool summary — Frozen vs TAR across all 56 candidate datasets.

Main paper: make_joint_tar_figure()  — Figure 3 (Frozen vs TAR, all / MulTaBench panels)
Appendix:   make_tfidf_figure()       — TF-IDF vs E5-Frozen vs E5-TAR
            make_struct_unstruct_figure() — Structured / Unstructured / Frozen / TAR
"""
import os
import numpy as np
import pandas as pd
import matplotlib

from multabench.leaderboard.data.keys import TEST_SCORE

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
_CORPUS  = os.path.join(_RESULTS, "tabstar_corpus")

_COLOR_TFIDF    = "#C8C8C8"
_COLOR_FROZEN   = "#A8D4F0"
_COLOR_TAR      = "#E8722A"
_COLOR_STRUCT   = "#98D98E"
_COLOR_UNSTRUCT = "#FFD06A"

_CI_TFIDF    = "#888888"
_CI_FROZEN   = "#3A88C8"
_CI_TAR      = "#B85010"
_CI_STRUCT   = "#3A9A30"
_CI_UNSTRUCT = "#C89000"

_FS_TITLE   = 15
_FS_YLABEL  = 15
_FS_XTICK   = 13
_FS_YTICK   = 13
_FS_LEGEND  = 13
_FONTWEIGHT = "normal"
_CAPSIZE    = 2
_ERR_LW     = 0.7

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM", "TabPFN-v2 🤯": "TabPFNv2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
}
_MODELS = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"]

# The 20 MulTaBench accepted text datasets (pool CSV naming).
_MULTABENCH_POOL_NAMES = frozenset({
    "BIN_TEXT_PROFESSIONAL_FAKE_JOB_POSTING",
    "BIN_TEXT_PROFESSIONAL_KICKSTARTER_FUNDING",
    "BIN_TEXT_SOCIAL_JIGSAW_TOXICITY",
    "MUL_TEXT_CONSUMER_PRODUCT_SENTIMENT",
    "MUL_TEXT_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW",
    "MUL_TEXT_FOOD_MICHELIN_GUIDE_RESTAURANTS",
    "MUL_TEXT_FOOD_WINE_REVIEW",
    "MUL_TEXT_PROFESSIONAL_DATA_SCIENTIST_SALARY",
    "MUL_TEXT_SOCIAL_SPOTIFY_GENRES",
    "MUL_TEXT_TRANSPORTATION_US_ACCIDENTS_MARCH23",
    "REG_TEXT_CONSUMER_BABIES_R_US_PRICES",
    "REG_TEXT_CONSUMER_BOOK_PRICE_PREDICTION",
    "REG_TEXT_CONSUMER_MERCARI_ONLINE_MARKETPLACE",
    "REG_TEXT_FOOD_ZOMATO_RESTAURANTS",
    "REG_TEXT_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER",
    "REG_TEXT_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY",
    "REG_TEXT_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT",
    "REG_TEXT_SOCIAL_BOOK_READABILITY_CLEAR",
    "REG_TEXT_SOCIAL_MOVIES_ROTTEN_TOMATOES",
    "REG_TEXT_SOCIAL_VIDEO_GAMES_SALES",
})

_STRUCT_STATE_MAP  = {"no_text": "Unimodal Structured", "text_only": "Unimodal Unstructured", "all": "Joint Frozen", "ft": "Joint TAR"}
_STRUCT_CONDITIONS = ["Unimodal Structured", "Unimodal Unstructured", "Joint Frozen", "Joint TAR"]
_STRUCT_COLORS     = [_COLOR_STRUCT, _COLOR_UNSTRUCT, _COLOR_FROZEN, _COLOR_TAR]
_STRUCT_CI_COLORS  = [_CI_STRUCT,   _CI_UNSTRUCT,    _CI_FROZEN,    _CI_TAR]

_JOINT_TAR_STATE_MAP  = {"all": "Frozen", "ft": "TAR"}
_JOINT_TAR_CONDITIONS = ["Frozen", "TAR"]
_JOINT_TAR_COLORS     = [_COLOR_FROZEN, _COLOR_TAR]
_JOINT_TAR_CI_COLORS  = [_CI_FROZEN,    _CI_TAR]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load() -> pd.DataFrame:
    df1 = pd.read_csv(os.path.join(_CORPUS, "text_50_datasets.csv"))
    df2 = pd.read_csv(os.path.join(_CORPUS, "texttabench_datasets.csv"))
    if "dataset_name" in df2.columns and "dataset" not in df2.columns:
        df2 = df2.rename(columns={"dataset_name": "dataset"})
    df = pd.concat([df1, df2], ignore_index=True)
    df["model"]       = df["model"].str.strip()
    df["test_score"]  = df["test_score"].clip(lower=-0.1)
    df["model_label"] = df["model"].map(_MODEL_LABELS)
    df = df[df["model_label"].notna()]
    return df[["dataset", "fold", "model_label", "multimodal_state", "test_score"]]


def _load_tfidf_df() -> pd.DataFrame:
    tfidf = pd.read_csv(os.path.join(_RESULTS, "analysis_tfidf/tfidf.csv"))
    tfidf["model"]      = tfidf["model"].str.strip()
    tfidf["test_score"] = tfidf["test_score"].clip(lower=-0.1)
    tfidf = tfidf[["dataset", "fold", "model", "test_score"]]
    tfidf_ds = set(tfidf["dataset"].unique())

    frames = []
    for state, label in [("all", "E5-small, Frozen"), ("ft", "E5-small, TAR")]:
        parts = []
        for f in os.listdir(os.path.join(_RESULTS, "text")):
            if not f.endswith(".csv"):
                continue
            sub = pd.read_csv(os.path.join(_RESULTS, "text", f))
            if "dataset" not in sub.columns:
                sub["dataset"] = f.replace(".csv", "")
            sub["model"]      = sub["model"].str.strip()
            sub["test_score"] = sub["test_score"].clip(lower=-0.1)
            parts.append(sub)
        e5 = pd.concat(parts, ignore_index=True)
        e5 = e5[e5["multimodal_state"] == state][["dataset", "fold", "model", "test_score"]]
        e5 = e5[e5["dataset"].isin(tfidf_ds)]
        e5["condition"] = label
        frames.append(e5)

    tfidf["condition"] = "TF-IDF"
    all_df = pd.concat([tfidf] + frames, ignore_index=True)
    counts = all_df.groupby("dataset")["condition"].nunique()
    valid_ds = counts[counts == 3].index
    all_df = all_df[all_df["dataset"].isin(valid_ds)].copy()
    all_df["model_label"] = all_df["model"].map(_MODEL_LABELS).fillna(
        all_df["model"].str.split().str[0])
    return all_df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _normalize_pool(df: pd.DataFrame, state_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["condition"] = df["multimodal_state"].map(state_map)
    df = df[df["condition"].notna()]
    lo = df.groupby(["dataset", "fold"])[TEST_SCORE].transform("min")
    hi = df.groupby(["dataset", "fold"])[TEST_SCORE].transform("max")
    df["norm"] = (df[TEST_SCORE] - lo) / (hi - lo).clip(lower=1e-9)
    agg = (df.groupby(["model_label", "condition"])["norm"]
             .agg(mean="mean", std="std", n="count")
             .reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5
    return agg


def _raw_pool(df: pd.DataFrame, state_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["condition"] = df["multimodal_state"].map(state_map)
    df = df[df["condition"].notna()]
    agg = (df.groupby(["model_label", "condition"])[TEST_SCORE]
             .agg(mean="mean", std="std", n="count")
             .reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5
    return agg


def _normalize_tfidf(df: pd.DataFrame) -> pd.DataFrame:
    conds = ["TF-IDF", "E5-small, Frozen", "E5-small, TAR"]
    df = df.copy()
    lo = df.groupby(["dataset", "fold", "model_label"])[TEST_SCORE].transform("min")
    hi = df.groupby(["dataset", "fold", "model_label"])[TEST_SCORE].transform("max")
    df["norm"] = (df["test_score"] - lo) / (hi - lo).clip(lower=1e-9)
    agg = (df.groupby(["model_label", "condition"])["norm"]
             .agg(mean="mean", std="std", n="count")
             .reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5
    agg["condition"] = pd.Categorical(agg["condition"], categories=conds, ordered=True)
    return agg.sort_values("condition")


# ---------------------------------------------------------------------------
# Shared panel drawing
# ---------------------------------------------------------------------------

def _draw_panel(ax, agg, conditions, colors, ci_colors, title,
                ylabel="Normalized Score", ylim=(0, 1.01), score_col="mean"):
    n_conds   = len(conditions)
    width     = 0.18 if n_conds >= 4 else 0.25
    group_gap = 0.25 if n_conds >= 4 else 0.20
    x_centers = np.arange(len(_MODELS)) * (n_conds * width + group_gap)
    offsets   = (np.arange(n_conds) - (n_conds - 1) / 2) * width

    for ci, (cond, color, ci_color) in enumerate(zip(conditions, colors, ci_colors)):
        for mi, model in enumerate(_MODELS):
            row = agg[(agg["model_label"] == model) & (agg["condition"] == cond)]
            if row.empty:
                continue
            x = x_centers[mi] + offsets[ci]
            ax.bar(x, row[score_col].iloc[0], width,
                   color=color, edgecolor="black", linewidth=0.6, zorder=2)
            ax.errorbar(x, row[score_col].iloc[0], yerr=row["ci"].iloc[0],
                        fmt="none", ecolor=ci_color, capsize=_CAPSIZE,
                        linewidth=_ERR_LW, zorder=3)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(_MODELS, fontsize=_FS_XTICK, fontweight=_FONTWEIGHT)
    ax.set_ylabel(ylabel, fontsize=_FS_YLABEL, fontweight=_FONTWEIGHT)
    if ylim is not None:
        ax.set_ylim(*ylim)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"],
                           fontsize=_FS_YTICK, fontweight=_FONTWEIGHT)
    else:
        ax.yaxis.set_tick_params(labelsize=_FS_YTICK)
        for tick in ax.get_yticklabels():
            tick.set_fontweight(_FONTWEIGHT)
    ax.set_title(title, fontsize=_FS_TITLE, fontweight=_FONTWEIGHT, pad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Public figure functions
# ---------------------------------------------------------------------------

def make_joint_tar_figure():
    """Figure 3 — Frozen vs TAR, All / MulTaBench panels (normalized)."""
    df    = _load()
    n_all = df["dataset"].nunique()
    df_mb = df[df["dataset"].isin(_MULTABENCH_POOL_NAMES)]
    n_mb  = df_mb["dataset"].nunique()

    agg_all = _normalize_pool(df,    _JOINT_TAR_STATE_MAP)
    agg_mb  = _normalize_pool(df_mb, _JOINT_TAR_STATE_MAP)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.2))
    fig.subplots_adjust(left=0.07, right=0.80, top=0.92, bottom=0.14, wspace=0.22)

    _draw_panel(axes[0], agg_all, _JOINT_TAR_CONDITIONS, _JOINT_TAR_COLORS, _JOINT_TAR_CI_COLORS,
                f"All ({n_all} datasets)")
    _draw_panel(axes[1], agg_mb,  _JOINT_TAR_CONDITIONS, _JOINT_TAR_COLORS, _JOINT_TAR_CI_COLORS,
                f"MulTaBench ({n_mb} datasets)", ylabel="")

    legend_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.6, label=l)
                      for c, l in zip(_JOINT_TAR_COLORS, _JOINT_TAR_CONDITIONS)]
    fig.legend(handles=legend_handles,
               loc="center left", bbox_to_anchor=(0.81, 0.5),
               frameon=True, edgecolor="black", framealpha=0.95,
               prop={"size": _FS_LEGEND})

    return fig, {"All": agg_all, "MulTaBench": agg_mb}


def make_tfidf_figure():
    """Appendix — TF-IDF vs E5-Frozen vs E5-TAR (normalized)."""
    TFIDF_CONDITIONS = ["TF-IDF", "E5-small, Frozen", "E5-small, TAR"]
    TFIDF_COLORS     = [_COLOR_TFIDF, _COLOR_FROZEN, _COLOR_TAR]
    TFIDF_CI_COLORS  = [_CI_TFIDF,   _CI_FROZEN,    _CI_TAR]

    df_all  = _load_tfidf_df()
    n_all   = df_all["dataset"].nunique()
    agg_all = _normalize_tfidf(df_all)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.2), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.82, top=0.92, bottom=0.14, wspace=0.12)

    _draw_panel(axes[0], agg_all, TFIDF_CONDITIONS, TFIDF_COLORS, TFIDF_CI_COLORS,
                f"All ({n_all} datasets with TF-IDF)")
    _draw_panel(axes[1], agg_all, TFIDF_CONDITIONS, TFIDF_COLORS, TFIDF_CI_COLORS,
                f"MulTaBench ({n_all} datasets)", ylabel="")

    legend_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.6, label=l)
                      for c, l in zip(TFIDF_COLORS, TFIDF_CONDITIONS)]
    fig.legend(handles=legend_handles,
               loc="center left", bbox_to_anchor=(0.83, 0.5),
               frameon=True, edgecolor="black", framealpha=0.95,
               prop={"size": _FS_LEGEND})

    return fig, agg_all


def make_struct_unstruct_figure():
    """Appendix — Structured / Unstructured / Frozen / TAR (normalized)."""
    df    = _load()
    n_all = df["dataset"].nunique()
    df_mb = df[df["dataset"].isin(_MULTABENCH_POOL_NAMES)]
    n_mb  = df_mb["dataset"].nunique()

    agg_all = _normalize_pool(df,    _STRUCT_STATE_MAP)
    agg_mb  = _normalize_pool(df_mb, _STRUCT_STATE_MAP)

    fig, axes = plt.subplots(1, 2, figsize=(18, 4.2), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.80, bottom=0.14, wspace=0.12)

    _draw_panel(axes[0], agg_all, _STRUCT_CONDITIONS, _STRUCT_COLORS, _STRUCT_CI_COLORS,
                f"All ({n_all} datasets)")
    _draw_panel(axes[1], agg_mb,  _STRUCT_CONDITIONS, _STRUCT_COLORS, _STRUCT_CI_COLORS,
                f"MulTaBench ({n_mb} datasets)", ylabel="")

    legend_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.6, label=l)
                      for c, l in zip(_STRUCT_COLORS, _STRUCT_CONDITIONS)]
    fig.legend(handles=legend_handles,
               loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=True, edgecolor="black", framealpha=0.95,
               prop={"size": _FS_LEGEND})

    return fig, {"All": agg_all, "MulTaBench": agg_mb}
