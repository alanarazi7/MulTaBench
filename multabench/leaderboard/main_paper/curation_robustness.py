"""Paper figure: how curation membership responds to the acceptance rule's two knobs.

(a) Datasets still admitted as the margin delta tightens, image and text separately, at each
    of three quorums. Computed over the 40 released datasets from benchmark_threshold_grid.csv.
(b) For each pool candidate, the share of all five-model panels drawn from 10 learners that
    would admit it. Computed over the 56-dataset text pool, the only subset where all 10
    learners have every condition, from committee_delta_sweep.csv.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_SENSITIVITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                            "analysis_curation_sensitivity")
_GRID_CSV = os.path.join(_SENSITIVITY, "benchmark_threshold_grid.csv")
_PANELS_CSV = os.path.join(_SENSITIVITY, "committee_delta_sweep.csv")

DELTA_DEFAULT = 0.001
DELTA_MAX = 0.02

_SUBSET_COLORS = {"image": "#1A5888", "text": "#C05010"}
_SUBSET_LABELS = {"image": "Image-Tabular", "text": "Text-Tabular"}
_RHO_STYLES = {0.6: "-", 0.8: "--", 1.0: ":"}
_RHO_LABELS = {0.6: r"$\rho = 3/5$", 0.8: r"$\rho = 4/5$", 1.0: r"$\rho = 5/5$"}

_DECISION_COLORS = {"accept": "#2A7A4B", "reject": "#8A8A8A"}

_FS_LABEL = 13
_FS_TICK = 11
_FS_LEGEND = 10
_FS_PANEL = 14


def _load_grid() -> pd.DataFrame:
    df = pd.read_csv(_GRID_CSV)
    df = df[(df["subset"] != "all") & (df["delta"] <= DELTA_MAX)]
    return df[df["rho"].isin([round(r, 3) for r in _RHO_STYLES])]


def _load_panels() -> pd.DataFrame:
    df = pd.read_csv(_PANELS_CSV)
    return df[df["delta"] == DELTA_DEFAULT]


def _draw_thresholds(ax, grid: pd.DataFrame):
    for subset, color in _SUBSET_COLORS.items():
        for rho, style in _RHO_STYLES.items():
            sub = (grid[(grid["subset"] == subset) & (grid["rho"] == round(rho, 3))]
                   .sort_values("delta"))
            ax.plot(sub["delta"], sub["n_surviving"], style, color=color,
                    linewidth=1.8, marker="o", markersize=3.5, zorder=3)

    ax.axvline(DELTA_DEFAULT, color="black", linewidth=1.0, linestyle="-", alpha=0.45, zorder=1)
    ax.annotate(r"MulTaBench $\delta$", xy=(DELTA_DEFAULT, 20.6), fontsize=_FS_LEGEND,
                ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel(r"Acceptance margin $\delta$", fontsize=_FS_LABEL)
    ax.set_ylabel("Datasets admitted (of 20)", fontsize=_FS_LABEL)
    ax.set_ylim(0, 22)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xticks([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02])
    ax.set_xticklabels(["0.0005", "0.001", "0.002", "0.005", "0.01", "0.02"], fontsize=_FS_TICK)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.minorticks_off()

    handles = [Line2D([], [], color=c, linewidth=2.2, label=_SUBSET_LABELS[s])
               for s, c in _SUBSET_COLORS.items()]
    handles += [Line2D([], [], linestyle="none", label=" ")]
    handles += [Line2D([], [], color="black", linestyle=st, linewidth=1.6, label=_RHO_LABELS[r])
                for r, st in _RHO_STYLES.items()]
    ax.legend(handles=handles, loc="lower left", ncol=2, frameon=True, edgecolor="black",
              framealpha=0.95, prop={"size": _FS_LEGEND})


def _draw_panel_rates(ax, panels: pd.DataFrame):
    bins = np.arange(0, 110, 10)
    groups = [panels[panels["original_decision"] == d]["pct_pass_ge3"]
              for d in ("reject", "accept")]
    ax.hist(groups, bins=bins, stacked=True,
            color=[_DECISION_COLORS["reject"], _DECISION_COLORS["accept"]],
            edgecolor="black", linewidth=0.6, zorder=2)

    ax.set_xlabel("Five-model panels admitting the dataset (%)", fontsize=_FS_LABEL)
    ax.set_ylabel("Pool candidates", fontsize=_FS_LABEL)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=_FS_TICK)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=_DECISION_COLORS[d], edgecolor="black",
                             linewidth=0.6, label=lbl)
               for d, lbl in [("accept", "Admitted to MulTaBench"), ("reject", "Rejected")]]
    ax.legend(handles=handles, loc="upper center", frameon=True, edgecolor="black",
              framealpha=0.95, prop={"size": _FS_LEGEND})


def make_figure():
    grid, panels = _load_grid(), _load_panels()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.17, wspace=0.24)

    _draw_thresholds(axes[0], grid)
    _draw_panel_rates(axes[1], panels)

    for ax, tag in zip(axes, ["(a)", "(b)"]):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(tag, fontsize=_FS_PANEL, loc="left")

    agg = {"(a) Threshold sensitivity": grid.reset_index(drop=True),
           "(b) Panel pass rates": panels.reset_index(drop=True)}
    return fig, agg
