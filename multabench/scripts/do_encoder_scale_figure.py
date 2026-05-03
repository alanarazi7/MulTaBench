"""Paper figure: encoder scale robustness (§6.1).

Replicates the 'Large Models' leaderboard chart with 'Normalize within model'
checked: for each (dataset, fold, model), the 4 conditions are rescaled to
[0, 1] by their own min/max, then averaged across datasets and folds.
Two panels side by side: DINO image (left) and E5 text (right).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE    = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "multimodal/leaderboard/results")
_OUT     = os.path.join(_HERE, "../paper-multabench/figures/encoder_scale.pdf")

# ---------------------------------------------------------------------------
# Shared style constants — must match paper_production.py / leaderboard figure
# ---------------------------------------------------------------------------
_COLOR_FROZEN_SMALL    = "#A8D4F0"
_COLOR_FINETUNED_SMALL = "#E8722A"
_COLOR_FROZEN_LARGE    = "#D0EEFF"
_COLOR_FINETUNED_LARGE = "#FBC9A0"
_HATCH_LARGE           = "\\\\"

_FS_TITLE      = 13
_FS_XLABEL     = 14
_FS_YTICK      = 12
_FS_XTICK      = 11
_FS_LEGEND     = 11
_FS_LEGTITLE   = 11
_FONTWEIGHT    = "bold"
_BAR_EDGE_DARK = "#555"
_CAPSIZE       = 2
_ERR_LW        = 0.7

# ---------------------------------------------------------------------------
# Condition labels
# ---------------------------------------------------------------------------
DINO_SMALL_ALL = "DINO-small (all)"
DINO_SMALL_FT  = "DINO-small (ft)"
DINO_LARGE_ALL = "DINO-large (all)"
DINO_LARGE_FT  = "DINO-large (ft)"
E5_SMALL_ALL   = "E5-small (all)"
E5_SMALL_FT    = "E5-small (ft)"
E5_LARGE_ALL   = "E5-large (all)"
E5_LARGE_FT    = "E5-large (ft)"

DINO_CONDITIONS = [DINO_SMALL_ALL, DINO_SMALL_FT, DINO_LARGE_ALL, DINO_LARGE_FT]
E5_CONDITIONS   = [E5_SMALL_ALL,   E5_SMALL_FT,   E5_LARGE_ALL,   E5_LARGE_FT]

# (color, hatch, edgecolor, label) per condition: small-all, small-ft, large-all, large-ft
_COND_STYLE = [
    (_COLOR_FROZEN_SMALL,    None,         "white",      "Small, Frozen"),
    (_COLOR_FINETUNED_SMALL, None,         "white",      "Small, Contextualized"),
    (_COLOR_FROZEN_LARGE,    _HATCH_LARGE, _BAR_EDGE_DARK, "Large, Frozen"),
    (_COLOR_FINETUNED_LARGE, _HATCH_LARGE, _BAR_EDGE_DARK, "Large, Contextualized"),
]

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM", "TabPFN-v2 🤯": "TabPFN-v2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-v2.5",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


def _build_dino_df() -> pd.DataFrame:
    large = _load_dir(os.path.join(_RESULTS, "images_large"), mm_filter=["all", "ft"])
    small = _load_dir(os.path.join(_RESULTS, "images"),      mm_filter=["all", "ft"])
    large_ds = set(large["dataset"].unique())
    small = small[small["dataset"].isin(large_ds)]
    small["mode"] = small["multimodal_state"].map({"all": DINO_SMALL_ALL, "ft": DINO_SMALL_FT})
    large["mode"] = large["multimodal_state"].map({"all": DINO_LARGE_ALL, "ft": DINO_LARGE_FT})
    return pd.concat([small[["dataset", "fold", "model", "mode", "test_score"]],
                      large[["dataset", "fold", "model", "mode", "test_score"]]],
                     ignore_index=True)


def _build_e5_df() -> pd.DataFrame:
    large = _load_dir(os.path.join(_RESULTS, "text_large"), mm_filter=["all", "ft"])
    small = _load_dir(os.path.join(_RESULTS, "text"),       mm_filter=["all", "ft"])
    large_ds = set(large["dataset"].unique())
    small = small[small["dataset"].isin(large_ds)]
    small["mode"] = small["multimodal_state"].map({"all": E5_SMALL_ALL, "ft": E5_SMALL_FT})
    large["mode"] = large["multimodal_state"].map({"all": E5_LARGE_ALL, "ft": E5_LARGE_FT})
    return pd.concat([small[["dataset", "fold", "model", "mode", "test_score"]],
                      large[["dataset", "fold", "model", "mode", "test_score"]]],
                     ignore_index=True)


# ---------------------------------------------------------------------------
# Normalization and aggregation
# ---------------------------------------------------------------------------

def _normalize_within_model(df: pd.DataFrame) -> pd.DataFrame:
    """Within each (dataset, fold, model), rescale the 4 conditions to [0, 1]."""
    df = df.copy()
    lo = df.groupby(["dataset", "fold", "model"])["test_score"].transform("min")
    hi = df.groupby(["dataset", "fold", "model"])["test_score"].transform("max")
    df["norm"] = (df["test_score"] - lo) / (hi - lo).clip(lower=1e-9)
    return df


def _aggregate(df: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["model_label"] = df["model"].map(_MODEL_LABELS).fillna(df["model"].str.split().str[0])
    agg = (df.groupby(["model_label", "mode"])["norm"]
             .agg(mean="mean", std="std", n="count")
             .reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5
    return agg[agg["mode"].isin(conditions)]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_panel(ax, agg: pd.DataFrame, conditions: list[str], title: str, n_datasets: int):
    models = sorted(agg["model_label"].unique())
    y      = np.arange(len(models))
    height = 0.18
    offset = (len(conditions) - 1) / 2

    for i, (cond, (color, hatch, edgecolor, label)) in enumerate(zip(conditions, _COND_STYLE)):
        subset = (agg[agg["mode"] == cond]
                  .set_index("model_label").reindex(models).reset_index())
        dy = (i - offset) * height * 1.15
        ax.barh(y + dy, subset["mean"], height,
                label=label, color=color, hatch=hatch,
                edgecolor=edgecolor, linewidth=0.8)
        ax.errorbar(subset["mean"], y + dy, xerr=subset["ci"],
                    fmt="none", ecolor="black", capsize=_CAPSIZE, linewidth=_ERR_LW)

    ax.set_title(f"{title} ({n_datasets} datasets)", fontsize=_FS_TITLE,
                 fontweight=_FONTWEIGHT, pad=7)
    ax.set_xlabel("Normalized Score", fontsize=_FS_XLABEL, fontweight=_FONTWEIGHT)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=_FS_YTICK, fontweight=_FONTWEIGHT)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=_FS_XTICK, fontweight=_FONTWEIGHT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_figure():
    dino_df = _build_dino_df()
    e5_df   = _build_e5_df()

    dino_norm = _normalize_within_model(dino_df)
    e5_norm   = _normalize_within_model(e5_df)

    dino_agg = _aggregate(dino_norm, DINO_CONDITIONS)
    e5_agg   = _aggregate(e5_norm,   E5_CONDITIONS)

    n_dino = dino_df["dataset"].nunique()
    n_e5   = e5_df["dataset"].nunique()

    from matplotlib.patches import Patch

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.subplots_adjust(wspace=0.45, left=0.10, right=0.84, top=0.90, bottom=0.12)

    _plot_panel(ax1, dino_agg, DINO_CONDITIONS, "DINO", n_dino)
    _plot_panel(ax2, e5_agg,   E5_CONDITIONS,   "E5",   n_e5)

    legend_handles = [
        Patch(facecolor=c, hatch=h, edgecolor=e, label=l)
        for c, h, e, l in _COND_STYLE
    ]
    fig.legend(handles=legend_handles, fontsize=_FS_LEGEND,
               title="Encoder size / mode", title_fontsize=_FS_LEGTITLE,
               loc="center left", bbox_to_anchor=(0.85, 0.5),
               frameon=True, edgecolor="none", framealpha=0.9)

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    fig.savefig(_OUT, format="pdf", dpi=200, bbox_inches="tight")
    print(f"Saved → {_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
