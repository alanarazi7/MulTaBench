"""Leaderboard tab: text-tabular pool Frozen vs TAR summary across all 56 datasets."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import streamlit as st

from multabench.leaderboard.data.keys import DATASET, FOLD, MODEL, MM, TEST_SCORE, E5_MODEL, E5_SMALL, IS_TUNED, DINO_MODEL, DINO_SMALL
from multabench.leaderboard.text_results import _load_pool_corpus_data

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM",
    "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM",
    "TabPFN-v2 🤯": "TabPFNv2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
}
_MODELS_ORDER = ["CatBoost", "LightGBM", "TabM", "TabPFNv2", "TabPFN-2.5"]

_COLOR_FROZEN = "#A8D4F0"
_COLOR_TAR    = "#E8722A"
_CI_FROZEN    = "#3A88C8"
_CI_TAR       = "#B85010"

FROZEN_LABEL = "Frozen"
TAR_LABEL    = "TAR"


def _prep_data() -> pd.DataFrame:
    df = _load_pool_corpus_data()
    df["model_label"] = df[MODEL].map(_MODEL_LABELS)
    df = df[df["model_label"].notna()]
    df["condition"] = df[MM].map({"all": FROZEN_LABEL, "ft": TAR_LABEL})
    df = df[df["condition"].notna()]
    return df[["model_label", DATASET, FOLD, "condition", TEST_SCORE]].copy()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize per (dataset, fold) across both conditions and all models."""
    lo = df.groupby([DATASET, FOLD])[TEST_SCORE].transform("min")
    hi = df.groupby([DATASET, FOLD])[TEST_SCORE].transform("max")
    df = df.copy()
    df["norm"] = (df[TEST_SCORE] - lo) / (hi - lo).clip(lower=1e-9)
    return df


def _plot_overview(df: pd.DataFrame, n_ds: int):
    agg = (df.groupby(["model_label", "condition"])["norm"]
             .agg(mean="mean", std="std", n="count")
             .reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5

    n_conds   = 2
    width     = 0.25
    group_gap = 0.20
    group_w   = n_conds * width
    x_centers = np.arange(len(_MODELS_ORDER)) * (group_w + group_gap)
    offsets   = (np.arange(n_conds) - (n_conds - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.subplots_adjust(left=0.10, right=0.72, top=0.90, bottom=0.14)

    for ci, (cond, color, ci_color) in enumerate(
        [(FROZEN_LABEL, _COLOR_FROZEN, _CI_FROZEN),
         (TAR_LABEL,    _COLOR_TAR,    _CI_TAR)]
    ):
        for mi, model in enumerate(_MODELS_ORDER):
            row = agg[(agg["model_label"] == model) & (agg["condition"] == cond)]
            if row.empty:
                continue
            x = x_centers[mi] + offsets[ci]
            ax.bar(x, row["mean"].iloc[0], width,
                   color=color, edgecolor="black", linewidth=0.6, zorder=2)
            ax.errorbar(x, row["mean"].iloc[0], yerr=row["ci"].iloc[0],
                        fmt="none", ecolor=ci_color, capsize=2,
                        linewidth=0.7, zorder=3)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(_MODELS_ORDER, fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Score", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.01)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"], fontsize=11, fontweight="bold")
    ax.set_title(f"Text-Tabular Pool: Frozen vs TAR ({n_ds} datasets)", fontsize=13, fontweight="bold", pad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=_COLOR_FROZEN, edgecolor="black", linewidth=0.6, label="Frozen (E5 all)"),
        Patch(facecolor=_COLOR_TAR,    edgecolor="black", linewidth=0.6, label="TAR (E5 ft)"),
    ]
    fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(0.73, 0.5),
               frameon=True, edgecolor="black", framealpha=0.95,
               prop={"weight": "bold", "size": 11})
    st.pyplot(fig)
    plt.close(fig)


def _win_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=[DATASET, FOLD, "model_label"], columns="condition", values=TEST_SCORE
    ).reset_index()
    rows = []
    for model in _MODELS_ORDER:
        sub = pivot[pivot["model_label"] == model][[FROZEN_LABEL, TAR_LABEL]].dropna()
        n    = len(sub)
        wins = (sub[TAR_LABEL] > sub[FROZEN_LABEL]).sum()
        rate = wins / n if n > 0 else float("nan")
        ci   = 1.96 * np.sqrt(rate * (1 - rate) / n) if n > 0 else float("nan")
        rows.append({
            "Model":    model,
            "Pairs":    n,
            "TAR wins": int(wins),
            "Win Rate": f"{rate*100:.1f}%",
            "95% CI":   f"± {ci*100:.1f}%",
        })
    return pd.DataFrame(rows).set_index("Model")


def _per_dataset_win_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate per (dataset, model): fraction of folds where TAR > Frozen.

    Columns: one per model + 'All models' aggregate.
    Values: 0.0–1.0 (fraction of folds won by TAR).
    """
    pivot = df.pivot_table(
        index=[DATASET, FOLD, "model_label"], columns="condition", values=TEST_SCORE
    ).reset_index()
    pivot["tar_wins"] = pivot[TAR_LABEL] > pivot[FROZEN_LABEL]

    rows = {}
    for (dataset, model), grp in pivot.groupby([DATASET, "model_label"]):
        rate = grp["tar_wins"].mean()
        rows.setdefault(dataset, {})[model] = round(rate, 2)

    result = pd.DataFrame(rows).T.reindex(columns=_MODELS_ORDER)
    result.index.name = DATASET

    # All-models aggregate per dataset (mean win rate across models × folds)
    all_pivot = pivot.groupby(DATASET)["tar_wins"].mean().round(2)
    result.insert(0, "All models", all_pivot)

    return result.sort_values("All models", ascending=False)


def display_text_pool_tar():
    st.title("📈 Text Pool: Frozen vs TAR")
    df_raw = _prep_data()
    n_ds = df_raw[DATASET].nunique()
    n_folds = df_raw[FOLD].nunique()
    st.caption(f"{n_ds} datasets · {n_folds} folds · {df_raw['model_label'].nunique()} models · "
               f"normalized per (dataset, fold) across Frozen and TAR.")

    df_norm = _normalize(df_raw)

    st.subheader("Overview: normalized Frozen vs TAR per model")
    _plot_overview(df_norm, n_ds)

    st.divider()
    st.subheader("Win rate: TAR > Frozen per model")
    st.caption("Fraction of (dataset, fold) pairs where E5-ft outperforms E5-all. 95% CI via normal approximation.")
    st.dataframe(_win_rate_table(df_raw), use_container_width=True)

    st.divider()
    st.subheader("Per-dataset win rate: TAR > Frozen")
    st.caption("Fraction of folds where TAR beats Frozen, per dataset and model. 'All models' averages across all 5 models × 5 folds.")
    wr_df = _per_dataset_win_rate_table(df_raw)
    st.dataframe(wr_df.style.background_gradient(cmap="RdYlGn", axis=None, vmin=0, vmax=1), use_container_width=True)
