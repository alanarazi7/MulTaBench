"""Paper figure: pairwise Cohen's kappa between the pool learners' accept/reject votes.

Lower triangle of the 10x10 matrix over the 56-dataset text pool, the only subset where all 10
learners have every condition. The five curation panel members are marked, and the cell for the
two TabPFN variants is outlined: the pair the bloc-vote concern is about. Chance-corrected, so
it is not inflated by the pool's overall acceptance rate the way raw agreement would be.
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
_KAPPA_CSV = os.path.join(_SENSITIVITY, "committee_pairwise_kappa.csv")

_BLOC_PAIR = ("TabPFNv2", "TabPFN-2.5")
_BLOC_COLOR = "#C05010"
_KAPPA_MIN, _KAPPA_MAX = 0.15, 0.65

_FS_LABEL = 12
_FS_TICK = 11
_FS_CELL = 10


def _load_kappa() -> pd.DataFrame:
    return pd.read_csv(_KAPPA_CSV, index_col=0)


def _draw_kappa(ax, kappa: pd.DataFrame):
    models = list(kappa.index)
    n = len(models)
    lower = np.where(np.tril(np.ones((n, n)), k=-1) == 1, kappa.values, np.nan)

    im = ax.imshow(lower, cmap="Blues", vmin=_KAPPA_MIN, vmax=_KAPPA_MAX)
    for i in range(n):
        for j in range(i):
            val = lower[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=_FS_CELL,
                    color="white" if val > 0.5 else "black")

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
    kappa = _load_kappa()

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    fig.subplots_adjust(left=0.19, right=0.88, top=0.99, bottom=0.22)
    im = _draw_kappa(ax, kappa)

    bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03,
                       ticks=[0.2, 0.3, 0.4, 0.5, 0.6])
    bar.set_label(r"Cohen's $\kappa$", fontsize=_FS_LABEL)
    bar.ax.tick_params(labelsize=_FS_TICK)

    ax.plot([], [], marker="s", linestyle="none", markersize=9, markerfacecolor="none",
            markeredgecolor=_BLOC_COLOR, markeredgewidth=2.0, label="Same model family")
    ax.legend(loc="upper right", frameon=True, edgecolor="black", framealpha=0.95,
              prop={"size": _FS_TICK}, handletextpad=0.6)

    return fig, {"Pairwise kappa": kappa}
