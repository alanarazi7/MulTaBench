"""Paper figure: pairwise agreement between the pool learners' accept/reject votes.

Full 10x10 matrix over the 56-dataset text pool, the only subset where all 10 learners have
every condition. Each cell is the share of candidates on which the two learners cast the same
vote; the five curation panel members are starred.
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

_PCT_MIN, _PCT_MAX = 50, 100
_DIAGONAL_COLOR = "#E8E8E8"

_FS_LABEL = 12
_FS_TICK = 11
_FS_CELL = 10


def _load_agreement() -> pd.DataFrame:
    return pd.read_csv(_AGREEMENT_CSV, index_col=0)


def _draw_agreement(ax, agreement: pd.DataFrame):
    models = list(agreement.index)
    n = len(models)
    values = agreement.values.astype(float)
    np.fill_diagonal(values, np.nan)

    im = ax.imshow(values, cmap="RdYlGn", vmin=_PCT_MIN, vmax=_PCT_MAX)
    im.cmap.set_bad(_DIAGONAL_COLOR)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center",
                        fontsize=_FS_CELL)

    labels = [f"{m} *" if m in CURATION_MODELS else m for m in models]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=_FS_TICK)
    ax.set_yticklabels(labels, fontsize=_FS_TICK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    return im


def make_figure():
    agreement = _load_agreement()

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    fig.subplots_adjust(left=0.19, right=0.88, top=0.99, bottom=0.22)
    im = _draw_agreement(ax, agreement)

    bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03,
                       ticks=[50, 60, 70, 80, 90, 100])
    bar.set_label("Agreement (%)", fontsize=_FS_LABEL)
    bar.ax.tick_params(labelsize=_FS_TICK)

    return fig, {"Pairwise agreement (%)": agreement}
