"""Paper figure: bootstrapped Bradley-Terry (Elo) rating of every Frozen / TAR competitor.

Reads the combined split of elo_frozen_vs_tar.csv, produced by
`python -m multabench.leaderboard.analysis.elo_leaderboard`. Each of the 10 embedding learners
appears twice, Frozen and TAR; TabSTAR and ConTextTab are end-to-end for text and split into
Frozen / TAR for image. Ratings are anchored so RandomForest (Frozen) = 1000, with 95% CIs
bootstrapped over datasets.
"""
import os
from os.path import join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

_SENSITIVITY = join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                    "analysis_curation_sensitivity")
_ELO_CSV = join(_SENSITIVITY, "elo_frozen_vs_tar.csv")

_COLOR_FROZEN = "#3A88C8"
_COLOR_TAR = "#E8722A"
_COLOR_E2E = "#9B72CF"
_ANCHOR_ELO = 1000
_VALUE_X = 2150

_CURATION_LEARNERS = {"LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5"}

_FS_TICK = 9
_FS_LABEL = 11
_FS_VALUE = 8.5


def _competitor(model: str) -> tuple[str, str, str]:
    """(display label, base learner, condition) for a raw competitor name."""
    if model.endswith(" (Frozen)") or model.endswith(" (TAR)"):
        base, cond = model.rsplit(" (", 1)
        return model, base, cond.rstrip(")")
    if "-" in model and model.count("-") == 2:
        base, modality, cond = model.split("-")
        cond = "End2End" if (modality == "Text" and cond == "Frozen") else cond
        return f"{base} ({cond}-{modality})", base, cond
    return model, model, "End2End"


def load_elo() -> pd.DataFrame:
    df = pd.read_csv(_ELO_CSV)
    df = df[df["split"] == "combined"].sort_values("elo", ascending=False).reset_index(drop=True)
    df[["label", "base", "condition"]] = df["model"].apply(
        lambda m: pd.Series(_competitor(m)))
    df["curation"] = df["base"].isin(_CURATION_LEARNERS)
    return df[["rank", "label", "base", "condition", "curation", "elo", "ci_low", "ci_high",
               "n_datasets"]]


def _draw_elo(ax, elo: pd.DataFrame):
    colors = {"Frozen": _COLOR_FROZEN, "TAR": _COLOR_TAR, "End2End": _COLOR_E2E}
    y = range(len(elo))

    ax.axvline(_ANCHOR_ELO, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    for i, row in elo.iterrows():
        color = colors[row["condition"]]
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=color, linewidth=1.6,
                solid_capstyle="round", alpha=0.55, zorder=2)
        ax.plot(row["elo"], i, "o", color=color, markersize=5.5, zorder=3)
        ax.text(_VALUE_X, i, f"{row['elo']:,}", ha="right", va="center", fontsize=_FS_VALUE)

    ax.set_yticks(list(y))
    ax.set_yticklabels([lab + ("$^{*}$" if cur else "")
                        for lab, cur in zip(elo["label"], elo["curation"])], fontsize=_FS_TICK)
    ax.invert_yaxis()
    ax.set_xlim(890, _VALUE_X + 15)
    ax.set_ylim(len(elo) - 0.4, -0.6)
    ax.set_xlabel("Elo rating (RandomForest Frozen anchored at 1000)", fontsize=_FS_LABEL)
    ax.set_xticks([1000, 1200, 1400, 1600, 1800, 2000])
    ax.tick_params(axis="x", labelsize=_FS_TICK)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def make_figure():
    elo = load_elo()

    fig, ax = plt.subplots(figsize=(7.0, 7.2))
    fig.subplots_adjust(left=0.30, right=0.99, top=0.95, bottom=0.07)
    _draw_elo(ax, elo)

    handles = [Line2D([], [], color=_COLOR_TAR, marker="o", linestyle="-", markersize=5.5,
                      label="Target-Aware"),
               Line2D([], [], color=_COLOR_FROZEN, marker="o", linestyle="-", markersize=5.5,
                      label="Frozen"),
               Line2D([], [], color=_COLOR_E2E, marker="o", linestyle="-", markersize=5.5,
                      label="End-to-End")]
    ax.legend(handles=handles, loc="upper left", fontsize=_FS_TICK, frameon=False)

    return fig, {"Elo leaderboard (combined, 40 datasets)": elo}
