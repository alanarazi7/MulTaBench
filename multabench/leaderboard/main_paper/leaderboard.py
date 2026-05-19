"""Paper figure: Tabular Learners Performance Analysis (§5, Figure 4).

Two-panel horizontal bar chart: (a) Image-Tabular, (b) Text-Tabular.
Each bar is a model; stacked orange TAR over blue Frozen; E2E models shown separately.
"""
import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import streamlit as st

_RESULTS_ROOT = join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

_COLOR_FROZEN    = "#A8D4F0"
_COLOR_FINETUNED = "#E8722A"
_COLOR_E2E       = "#9B72CF"

_CORE_MODELS = {
    "LightGBM 💡", "CatBoost 😸", "TabM Ⓜ️",
    "TabPFN-v2 🤯", "TabPFN-v2p5 🇩🇪",
}
_E2E_IMAGE_MODELS = {"AutoGluon-MM 🧴"}
_E2E_TEXT_MODELS  = {"TabSTAR ⭐", "ConTextTab 🏢", "AutoGluon-MM 🧴"}

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM", "TabPFN-v2 🤯": "TabPFNv2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
    "TabSTAR ⭐": "TabSTAR", "ConTextTab 🏢": "ConTextTab",
    "AutoGluon-MM 🧴": "AG-MM",
    "TabDPT 6️⃣": "TabDPT", "TabICLv2 🗼": "TabICLv2",
    "RandomForest 🌳": "RF", "RealMLP 🕸": "RealMLP",
    "XGBoost 🌲": "XGBoost",
}


def _load_dir(subdir: str) -> pd.DataFrame:
    path = join(_RESULTS_ROOT, subdir)
    frames = []
    for f in os.listdir(path):
        if not f.endswith(".csv"):
            continue
        df = pd.read_csv(join(path, f))
        if "dataset" not in df.columns:
            df["dataset"] = f.replace(".csv", "")
        df["model"]      = df["model"].str.strip()
        df["test_score"] = df["test_score"].clip(lower=-0.1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def load_for_overview(modality: str, task_type: str = "all") -> pd.DataFrame:
    subdir = "images" if modality == "IMAGE" else "text"
    frames = [_load_dir(subdir), _load_dir("more_baselines")]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["dataset"].str.contains(f"_{modality}_")]
    if task_type == "cls":
        df = df[df["dataset"].str.startswith(("BIN_", "MUL_"))]
    elif task_type == "reg":
        df = df[df["dataset"].str.startswith("REG_")]
    return df[df["multimodal_state"].isin(["all", "ft"])]


def _plot_overview_ax(ax, df_raw: pd.DataFrame, modality: str, title: str, legend: bool = False):
    df = df_raw.copy()
    df["label"] = df["model"].map(_MODEL_LABELS).fillna(df["model"])
    e2e_raw    = _E2E_IMAGE_MODELS if modality == "IMAGE" else _E2E_TEXT_MODELS
    e2e_labels = {_MODEL_LABELS.get(m, m) for m in e2e_raw}

    non_e2e = df[~df["label"].isin(e2e_labels)]
    coverage = non_e2e.pivot_table(index="dataset", columns=["label", "multimodal_state"],
                                   values="test_score", aggfunc="count")
    complete_datasets = coverage.dropna(how="any").index
    df = df[df["dataset"].isin(complete_datasets)]

    lo = df.groupby(["dataset", "fold"])["test_score"].transform("min")
    hi = df.groupby(["dataset", "fold"])["test_score"].transform("max")
    df["norm"] = (df["test_score"] - lo) / (hi - lo).clip(lower=1e-9)

    agg = (df.groupby(["label", "multimodal_state"])["norm"]
             .agg(mean="mean", std="std", n="count").reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5

    all_mean  = agg[agg["multimodal_state"] == "all"].set_index("label")["mean"]
    ft_mean   = agg[agg["multimodal_state"] == "ft"].set_index("label")["mean"]
    models    = sorted(agg["label"].unique(),
                       key=lambda m: ft_mean.get(m, all_mean.get(m, 0)),
                       reverse=True)

    y      = np.arange(len(models)) * 0.82
    bar_h  = 0.55
    all_agg = agg[agg["multimodal_state"] == "all"].set_index("label")
    ft_agg  = agg[agg["multimodal_state"] == "ft"].set_index("label")
    core_labels = {_MODEL_LABELS.get(m, m) for m in _CORE_MODELS}

    for i, m in enumerate(models):
        if m in e2e_labels:
            if m not in all_agg.index or not pd.notna(all_agg.loc[m]["mean"]):
                continue
            row = all_agg.loc[m]
            ax.barh(y[i], row["mean"], bar_h, color=_COLOR_E2E,
                    edgecolor="#555", linewidth=0.6, zorder=2)
            ax.errorbar(row["mean"], y[i], xerr=row["ci"],
                        fmt="none", ecolor="black", capsize=2, linewidth=0.7, zorder=3)
        else:
            all_val = all_agg.loc[m]["mean"] if m in all_agg.index else np.nan
            all_ci  = all_agg.loc[m]["ci"]   if m in all_agg.index else np.nan
            ft_val  = ft_agg.loc[m]["mean"]  if m in ft_agg.index  else np.nan
            ft_ci   = ft_agg.loc[m]["ci"]    if m in ft_agg.index  else np.nan

            if not pd.notna(all_val):
                continue
            if pd.notna(ft_val):
                ax.barh(y[i], ft_val, bar_h, color=_COLOR_FINETUNED,
                        edgecolor="black", linewidth=0.6, zorder=2)
                ax.errorbar(ft_val, y[i], xerr=ft_ci,
                            fmt="none", ecolor="#B85010", capsize=2, linewidth=0.9, zorder=3)
            ax.barh(y[i], all_val, bar_h, color=_COLOR_FROZEN,
                    edgecolor="black", linewidth=0.6, zorder=3)
            ax.errorbar(all_val, y[i], xerr=all_ci,
                        fmt="none", ecolor="#3A88C8", capsize=2, linewidth=0.9, zorder=4)
            if m in core_labels:
                ax.text(0.02, y[i], "★", va="center", ha="left",
                        fontsize=9, color="black", zorder=5)

    ax.set_xlabel("Normalized Score", fontsize=15)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=13)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"],
                       fontsize=13)
    ax.set_ylim(y[0] - bar_h * 1.1, y[-1] + bar_h * 1.1)
    ax.set_title(title, fontsize=15, pad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    if legend:
        ax.legend(handles=[
            Patch(color=_COLOR_FROZEN,    edgecolor="black", label="Frozen"),
            Patch(color=_COLOR_FINETUNED, edgecolor="black", label="TAR"),
            Patch(facecolor=_COLOR_E2E,   edgecolor="#555",  label="E2E"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
                   markersize=11, label="Curation"),
        ], frameon=True, loc="upper right",
           framealpha=0.95, edgecolor="black",
           bbox_to_anchor=(1.0, 0.93), borderaxespad=0,
           prop={"size": 13})

    return agg


def make_figure(task_type: str = "all"):
    df_img = load_for_overview("IMAGE", task_type)
    df_txt = load_for_overview("TEXT",  task_type)

    n_img  = df_img["model"].nunique()
    n_txt  = df_txt["model"].nunique()
    height = max(n_img, n_txt) * 0.82 * 0.38 + 0.8

    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(10.5, max(4.5, height)))
    agg_img = _plot_overview_ax(ax_img, df_img, "IMAGE", "(a) Image-Tabular", legend=True)
    agg_txt = _plot_overview_ax(ax_txt, df_txt, "TEXT",  "(b) Text-Tabular",  legend=False)
    plt.tight_layout(w_pad=2.5)
    return fig, {"Image-Tabular": agg_img, "Text-Tabular": agg_txt}
