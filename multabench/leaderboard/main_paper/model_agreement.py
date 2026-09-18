"""Paper figure: pairwise agreement between the pool learners' accept/reject votes.

Lower triangle of the 10x10 matrix over the 56-dataset text pool, the only subset where all 10
learners have every condition. Each cell is the share of candidates on which the two learners
cast the same vote. The five curation panel members are marked, and the cell for the two TabPFN
variants is outlined: the pair the bloc-vote concern is about.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multabench.leaderboard.analysis.committee_pool import CURATION_MODELS

_SENSITIVITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                            "analysis_curation_sensitivity")
_AGREEMENT_CSV = os.path.join(_SENSITIVITY, "model_agreement_percent.csv")

_BLOC_PAIR = ("TabPFNv2", "TabPFN-2.5")
_BLOC_COLOR = "#C05010"
_PCT_MIN, _PCT_MAX = 55, 85

_FS_LABEL = 12
_FS_TICK = 11
_FS_CELL = 10


def _load_agreement() -> pd.DataFrame:
    return pd.read_csv(_AGREEMENT_CSV, index_col=0)


def _draw_agreement(ax, agreement: pd.DataFrame):
    models = list(agreement.index)
    n = len(models)
    lower = np.where(np.tril(np.ones((n, n)), k=-1) == 1, agreement.values, np.nan)

    im = ax.imshow(lower, cmap="Blues", vmin=_PCT_MIN, vmax=_PCT_MAX)
    for i in range(n):
        for j in range(i):
            val = lower[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=_FS_CELL,
                    color="white" if val > 76 else "black")

    bi, bj = sorted(models.index(m) for m in _BLOC_PAIR)[::-1]
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor=_BLOC_COLOR, linewidth=2.4, zorder=4))

    labels = [f"{m} *" if m in CURATION_MODELS else m for m in models]
    ax.set_xticks(range(n - 1))
    ax.set_yticks(range(1, n))
    ax.set_xticklabels(labels[:-1], rotation=45, ha="right", fontsize=_FS_TICK)
    ax.set_yticklabels(labels[1:], fontsize=_FS_TICK)
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    return im


def make_figure():
    agreement = _load_agreement()

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    fig.subplots_adjust(left=0.19, right=0.88, top=0.99, bottom=0.22)
    im = _draw_agreement(ax, agreement)

    bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03,
                       ticks=[55, 60, 65, 70, 75, 80, 85])
    bar.set_label("Agreement (%)", fontsize=_FS_LABEL)
    bar.ax.tick_params(labelsize=_FS_TICK)

    ax.plot([], [], marker="s", linestyle="none", markersize=9, markerfacecolor="none",
            markeredgecolor=_BLOC_COLOR, markeredgewidth=2.0, label="Same model family")
    ax.legend(loc="upper right", frameon=True, edgecolor="black", framealpha=0.95,
              prop={"size": _FS_TICK}, handletextpad=0.6)

    return fig, {"Pairwise agreement (%)": agreement}
