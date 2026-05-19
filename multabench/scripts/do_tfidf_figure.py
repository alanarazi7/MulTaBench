"""Paper figure: TF-IDF vs E5 text representation robustness (§6.1).

Three conditions: TF-IDF (frozen), E5-small (frozen), E5-small (contextualized),
on the 20 text datasets that have TF-IDF results. Grouped vertical bars per model,
normalized within each (dataset, fold, model) group across all 3 conditions.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_HERE    = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "..", "leaderboard", "results")
_OUT     = os.path.join(_HERE, "../../../paper-multabench/figures/tfidf.pdf")

# ---------------------------------------------------------------------------
# Style constants — match paper_production / encoder_scale / pca figures
# ---------------------------------------------------------------------------
_COLOR_TFIDF   = "#C8C8C8"
_COLOR_FROZEN  = "#A8D4F0"
_COLOR_CTX     = "#E8722A"

_FS_TITLE   = 15
_FS_YLABEL  = 15
_FS_XTICK   = 13
_FS_YTICK   = 13
_FS_LEGEND  = 13
_FONTWEIGHT = "normal"
_CAPSIZE    = 2
_ERR_LW     = 0.7

CONDITIONS = ["TF-IDF", "E5-small, Frozen", "E5-small, Contextualized"]
_COLORS    = [_COLOR_TFIDF, _COLOR_FROZEN, _COLOR_CTX]

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost",
    "TabM Ⓜ️": "TabM", "TabPFN-v2 🤯": "TabPFN-v2",
    "TabPFN-v2p5 🇩🇪": "TabPFN-v2.5",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dir_state(path: str, state: str) -> pd.DataFrame:
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
    df = pd.concat(frames, ignore_index=True)
    return df[df["multimodal_state"] == state][["dataset", "fold", "model", "test_score"]]


def _build_df() -> pd.DataFrame:
    tfidf = pd.read_csv(os.path.join(_RESULTS, "analysis_tfidf/tfidf.csv"))
    tfidf["model"]      = tfidf["model"].str.strip()
    tfidf["test_score"] = tfidf["test_score"].clip(lower=-0.1)
    tfidf = tfidf[["dataset", "fold", "model", "test_score"]]
    tfidf_ds = set(tfidf["dataset"].unique())

    e5_frozen = _load_dir_state(os.path.join(_RESULTS, "text"), "all")
    e5_ctx    = _load_dir_state(os.path.join(_RESULTS, "text"), "ft")
    e5_frozen = e5_frozen[e5_frozen["dataset"].isin(tfidf_ds)]
    e5_ctx    = e5_ctx[e5_ctx["dataset"].isin(tfidf_ds)]

    tfidf["condition"]     = "TF-IDF"
    e5_frozen["condition"] = "E5-small, Frozen"
    e5_ctx["condition"]    = "E5-small, Contextualized"

    combined = pd.concat([tfidf, e5_frozen, e5_ctx], ignore_index=True)
    counts = combined.groupby("dataset")["condition"].nunique()
    valid_ds = counts[counts == 3].index
    return combined[combined["dataset"].isin(valid_ds)]


# ---------------------------------------------------------------------------
# Normalization and aggregation
# ---------------------------------------------------------------------------

def _normalize_and_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_label"] = df["model"].map(_MODEL_LABELS).fillna(df["model"].str.split().str[0])
    lo = df.groupby(["dataset", "fold", "model_label"])["test_score"].transform("min")
    hi = df.groupby(["dataset", "fold", "model_label"])["test_score"].transform("max")
    df["norm"] = (df["test_score"] - lo) / (hi - lo).clip(lower=1e-9)
    agg = (df.groupby(["model_label", "condition"])["norm"]
             .agg(mean="mean", std="std", n="count")
             .reset_index())
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5
    return agg


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_figure():
    df  = _build_df()
    agg = _normalize_and_agg(df)
    n_ds = df["dataset"].nunique()

    models   = ["CatBoost", "LightGBM", "TabM", "TabPFN-v2", "TabPFN-v2.5"]
    n_conds  = len(CONDITIONS)
    width    = 0.18
    group_gap = 0.25
    group_w  = n_conds * width
    x_centers = np.arange(len(models)) * (group_w + group_gap)
    offsets   = (np.arange(n_conds) - (n_conds - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(left=0.08, right=0.72, top=0.90, bottom=0.14)

    for ci, (cond, color) in enumerate(zip(CONDITIONS, _COLORS)):
        for mi, model in enumerate(models):
            row = agg[(agg["model_label"] == model) & (agg["condition"] == cond)]
            if row.empty:
                continue
            x = x_centers[mi] + offsets[ci]
            ax.bar(x, row["mean"].iloc[0], width,
                   color=color, edgecolor="white", linewidth=0.6, zorder=2)
            ax.errorbar(x, row["mean"].iloc[0], yerr=row["ci"].iloc[0],
                        fmt="none", ecolor="black", capsize=_CAPSIZE,
                        linewidth=_ERR_LW, zorder=3)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(models, fontsize=_FS_XTICK, fontweight=_FONTWEIGHT)
    ax.set_ylabel("Normalized Score", fontsize=_FS_YLABEL, fontweight=_FONTWEIGHT)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=_FS_YTICK, fontweight=_FONTWEIGHT)
    ax.set_title(f"Text Representation ({n_ds} datasets)", fontsize=_FS_TITLE,
                 fontweight=_FONTWEIGHT, pad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    legend_handles = [Patch(facecolor=c, edgecolor="white", label=l)
                      for c, l in zip(_COLORS, CONDITIONS)]
    fig.legend(handles=legend_handles, fontsize=_FS_LEGEND,
               title="Text encoding", title_fontsize=_FS_LEGEND,
               loc="center left", bbox_to_anchor=(0.73, 0.5),
               frameon=True, edgecolor="none", framealpha=0.9)

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    fig.savefig(_OUT, format="pdf", dpi=200, bbox_inches="tight")
    print(f"Saved → {_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
