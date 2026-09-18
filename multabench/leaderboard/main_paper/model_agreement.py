"""Paper figure: pairwise agreement between the pool learners' accept/reject votes.

Full 10x10 matrix over the 56-dataset text pool, the only subset where all 10 learners have
every condition. Each cell is the share of candidates on which the two learners cast the same
vote. The colour scale starts at 50 to keep the observed spread legible.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SENSITIVITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                            "analysis_curation_sensitivity")
_AGREEMENT_CSV = os.path.join(_SENSITIVITY, "model_agreement_percent.csv")

_PCT_MIN, _PCT_MAX = 50, 100
_DIAGONAL_COLOR = "#E8E8E8"

_FS_LABEL = 12
_FS_TICK = 11
_FS_CELL = 10
_FS_XTICK = 9.5


def _load_agreement() -> pd.DataFrame:
    return pd.read_csv(_AGREEMENT_CSV, index_col=0)


def _wrap(model: str) -> str:
    """Two-line x labels so the names stay horizontal without colliding."""
    return {"LightGBM": "Light\nGBM", "CatBoost": "Cat\nBoost", "TabPFNv2": "TabPFN\nv2",
            "TabPFN-2.5": "TabPFN\n2.5", "RandomForest": "Random\nForest",
            "RealMLP": "Real\nMLP", "TabICLv2": "TabICL\nv2",
            "XGBoost": "XG\nBoost"}.get(model, model)


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

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([_wrap(m) for m in models], fontsize=_FS_XTICK)
    ax.set_yticklabels(models, fontsize=_FS_TICK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    return im


def make_figure():
    agreement = _load_agreement()

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    fig.subplots_adjust(left=0.19, right=0.88, top=0.99, bottom=0.22)
    im = _draw_agreement(ax, agreement)

    bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03,
                       ticks=[50, 60, 70, 80, 90, 100])
    bar.set_label("Agreement (%)", fontsize=_FS_LABEL)
    bar.ax.tick_params(labelsize=_FS_TICK)

    return fig, {"Pairwise agreement (%)": agreement}
