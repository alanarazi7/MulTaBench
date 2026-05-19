"""Paper figure: no-PCA ablation (§6.1).

Four conditions: 30 dims Frozen, 30 dims Contextualized (default), no-PCA Frozen,
no-PCA Contextualized (raw 384-dim embeddings). Restricted to LightGBM and CatBoost
on the 33 datasets covered by the no-PCA run. Grouped vertical bars per model.
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
_PCA_DIR = os.path.join(_RESULTS, "analysis_pca")
_OUT     = os.path.join(_HERE, "../../../paper-multabench/figures/nopca.pdf")

# ---------------------------------------------------------------------------
# Style constants — match paper_production / encoder_scale / pca figures
# ---------------------------------------------------------------------------
_COLOR_FROZEN     = "#A8D4F0"
_COLOR_FROZEN_NP  = "#D0EEFF"
_COLOR_CTX        = "#E8722A"
_COLOR_CTX_NP     = "#FBC9A0"
_HATCH_NOPCA      = "\\\\"

_FS_TITLE   = 15
_FS_YLABEL  = 15
_FS_XTICK   = 13
_FS_YTICK   = 13
_FS_LEGEND  = 13
_FONTWEIGHT = "normal"
_CAPSIZE    = 2
_ERR_LW     = 0.7

CONDITIONS = ["30 dims, Frozen", "30 dims, Contextualized",
              "No PCA, Frozen",  "No PCA, Contextualized"]
_COND_STYLE = [
    (_COLOR_FROZEN,    None,        "white"),
    (_COLOR_CTX,       None,        "white"),
    (_COLOR_FROZEN_NP, _HATCH_NOPCA, "#555"),
    (_COLOR_CTX_NP,    _HATCH_NOPCA, "#555"),
]

_MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost",
}
_MODELS = ["CatBoost", "LightGBM"]

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
    nopca = pd.read_csv(os.path.join(_PCA_DIR, "no_pca.csv"))
    nopca["model"]      = nopca["model"].str.strip()
    nopca["test_score"] = nopca["test_score"].clip(lower=-0.1)
    nopca_ds = set(nopca["dataset"].unique())

    img_all = _load_dir_state(os.path.join(_RESULTS, "images"), "all")
    txt_all = _load_dir_state(os.path.join(_RESULTS, "text"),   "all")
    img_ft  = _load_dir_state(os.path.join(_RESULTS, "images"), "ft")
    txt_ft  = _load_dir_state(os.path.join(_RESULTS, "text"),   "ft")
    d30_all = pd.concat([img_all, txt_all], ignore_index=True)
    d30_ft  = pd.concat([img_ft,  txt_ft],  ignore_index=True)

    for df in [d30_all, d30_ft]:
        df["model"] = df["model"].map(_MODEL_LABELS).fillna(df["model"])

    lgbm_cat = set(_MODEL_LABELS.values())

    pieces = {
        "30 dims, Frozen":         d30_all[d30_all["dataset"].isin(nopca_ds) & d30_all["model"].isin(lgbm_cat)],
        "30 dims, Contextualized": d30_ft[d30_ft["dataset"].isin(nopca_ds)   & d30_ft["model"].isin(lgbm_cat)],
        "No PCA, Frozen":          nopca[nopca["multimodal_state"] == "all"][["dataset", "fold", "model", "test_score"]],
        "No PCA, Contextualized":  nopca[nopca["multimodal_state"] == "ft"][["dataset",  "fold", "model", "test_score"]],
    }
    for df in pieces.values():
        df["model"] = df["model"].map(_MODEL_LABELS).fillna(df["model"])

    frames = []
    for cond, df in pieces.items():
        df = df.copy()
        df["condition"] = cond
        frames.append(df[["dataset", "fold", "model", "test_score", "condition"]])
    combined = pd.concat(frames, ignore_index=True)
    counts = combined.groupby("dataset")["condition"].nunique()
    valid_ds = counts[counts == 4].index
    return combined[combined["dataset"].isin(valid_ds)]


# ---------------------------------------------------------------------------
# Normalization and aggregation
# ---------------------------------------------------------------------------

def _normalize_and_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lo = df.groupby(["dataset", "fold", "model"])["test_score"].transform("min")
    hi = df.groupby(["dataset", "fold", "model"])["test_score"].transform("max")
    df["norm"] = (df["test_score"] - lo) / (hi - lo).clip(lower=1e-9)
    agg = (df.groupby(["model", "condition"])["norm"]
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

    n_conds   = len(CONDITIONS)
    width     = 0.18
    group_gap = 0.25
    group_w   = n_conds * width
    x_centers = np.arange(len(_MODELS)) * (group_w + group_gap)
    offsets   = (np.arange(n_conds) - (n_conds - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.subplots_adjust(left=0.12, right=0.65, top=0.90, bottom=0.14)

    for ci, (cond, (color, hatch, edgecolor)) in enumerate(zip(CONDITIONS, _COND_STYLE)):
        for mi, model in enumerate(_MODELS):
            row = agg[(agg["model"] == model) & (agg["condition"] == cond)]
            if row.empty:
                continue
            x = x_centers[mi] + offsets[ci]
            ax.bar(x, row["mean"].iloc[0], width,
                   color=color, hatch=hatch, edgecolor=edgecolor,
                   linewidth=0.6, zorder=2)
            ax.errorbar(x, row["mean"].iloc[0], yerr=row["ci"].iloc[0],
                        fmt="none", ecolor="black", capsize=_CAPSIZE,
                        linewidth=_ERR_LW, zorder=3)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(_MODELS, fontsize=_FS_XTICK, fontweight=_FONTWEIGHT)
    ax.set_ylabel("Normalized Score", fontsize=_FS_YLABEL, fontweight=_FONTWEIGHT)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=_FS_YTICK, fontweight=_FONTWEIGHT)
    ax.set_title(f"No PCA Ablation ({n_ds} datasets)", fontsize=_FS_TITLE,
                 fontweight=_FONTWEIGHT, pad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=c, hatch=h, edgecolor=e, label=l)
        for l, (c, h, e) in zip(CONDITIONS, _COND_STYLE)
    ]
    fig.legend(handles=legend_handles, fontsize=_FS_LEGEND,
               title="PCA / mode", title_fontsize=_FS_LEGEND,
               loc="center left", bbox_to_anchor=(0.66, 0.5),
               frameon=True, edgecolor="none", framealpha=0.9)

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    fig.savefig(_OUT, format="pdf", dpi=200, bbox_inches="tight")
    print(f"Saved → {_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
