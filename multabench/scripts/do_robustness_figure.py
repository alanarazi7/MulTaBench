"""Generate 3-panel robustness figure for §6.1 of the MulTaBench paper.

Panel (a) and (b) show the FT gap (ft_score − frozen_score) per condition,
averaged across (dataset, fold, model). This is metric-scale-invariant since
we compare ft vs frozen on the *same* datasets, so AUROC/R² differences cancel.

Panel (c) shows per-dataset normalized scores [0,1] within {TF-IDF, E5-frozen,
E5-ft}, removing metric-scale differences across text datasets.
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
_OUT     = os.path.join(_HERE, "../../../paper-multabench/figures/robustness.pdf")

_COLOR_FROZEN    = "#A8D4F0"
_COLOR_FINETUNED = "#E8722A"
_COLOR_TFIDF     = "#C8C8C8"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_dir(path: str) -> pd.DataFrame:
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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_pca_csv(fpath: str, mode: str) -> pd.DataFrame:
    df = pd.read_csv(fpath)
    if "dataset_name" in df.columns and "dataset" not in df.columns:
        df = df.rename(columns={"dataset_name": "dataset"})
    df["model"]            = df["model"].str.strip()
    df["test_score"]       = df["test_score"].clip(lower=-0.1)
    df["multimodal_state"] = mode
    return df[["dataset", "fold", "model", "multimodal_state", "test_score"]]


# ---------------------------------------------------------------------------
# FT-gap: mean(ft_score − frozen_score) per (dataset, fold), averaged across
# (dataset, fold), with 95% CI across datasets.
# ---------------------------------------------------------------------------

def _ft_gap(df_frozen: pd.DataFrame, df_ft: pd.DataFrame, label: str) -> dict:
    """Compute mean FT gap per (dataset, fold, model), then average."""
    key = ["dataset", "fold", "model"]
    m_all = df_frozen[key + ["test_score"]].rename(columns={"test_score": "score_all"})
    m_ft  = df_ft[key + ["test_score"]].rename(columns={"test_score": "score_ft"})
    merged = m_all.merge(m_ft, on=key)
    merged["gap"] = merged["score_ft"] - merged["score_all"]
    # Average across models first, then across (dataset, fold)
    per_ds_fold = merged.groupby(["dataset", "fold"])["gap"].mean()
    mean_gap = per_ds_fold.mean()
    n        = len(per_ds_fold)
    ci       = 1.96 * per_ds_fold.std() / np.sqrt(n) if n > 1 else 0.0
    return {"label": label, "mean": mean_gap, "ci": ci}


# ---------------------------------------------------------------------------
# Per-dataset normalization for TF-IDF panel
# ---------------------------------------------------------------------------

def _normalize_within(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize per (dataset, fold) using min/max across all rows."""
    df = df.copy()
    lo = df.groupby(["dataset", "fold"])["test_score"].transform("min")
    hi = df.groupby(["dataset", "fold"])["test_score"].transform("max")
    df["norm"] = (df["test_score"] - lo) / (hi - lo).clip(lower=1e-9)
    return df


def _agg_norm(df: pd.DataFrame, condition_col: str) -> pd.DataFrame:
    per_ds_fold = df.groupby(["dataset", "fold", condition_col])["norm"].mean().reset_index()
    agg = per_ds_fold.groupby(condition_col)["norm"].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    agg["ci"] = 1.96 * agg["std"] / agg["n"] ** 0.5
    return agg


# ---------------------------------------------------------------------------
# Panel data
# ---------------------------------------------------------------------------

def _encoder_data():
    img_small = _load_dir(os.path.join(_RESULTS, "images"))
    img_large = _load_dir(os.path.join(_RESULTS, "images_large"))
    txt_small = _load_dir(os.path.join(_RESULTS, "text"))
    txt_large = _load_dir(os.path.join(_RESULTS, "text_large"))

    img_ds = set(img_large["dataset"].unique())
    txt_ds = set(txt_large["dataset"].unique())

    def _filt(df, ds, state):
        return df[df["dataset"].isin(ds) & (df["multimodal_state"] == state)]

    rows = [
        _ft_gap(_filt(img_small, img_ds, "all"), _filt(img_small, img_ds, "ft"), "DINO-small\n(image)"),
        _ft_gap(_filt(img_large, img_ds, "all"), _filt(img_large, img_ds, "ft"), "DINO-large\n(image)"),
        _ft_gap(_filt(txt_small, txt_ds, "all"), _filt(txt_small, txt_ds, "ft"), "E5-small\n(text)"),
        _ft_gap(_filt(txt_large, txt_ds, "all"), _filt(txt_large, txt_ds, "ft"), "E5-large\n(text)"),
    ]
    return pd.DataFrame(rows)


def _pca_data():
    pca_dir = os.path.join(_RESULTS, "analysis_pca")

    # Get common datasets from the 15-dim ablation
    base_ds = set(_load_pca_csv(os.path.join(pca_dir, "all_15.csv"), "all")["dataset"].unique())

    rows = []
    for dim, label in [(15, "15 dims"), (60, "60 dims")]:
        df_all = _load_pca_csv(os.path.join(pca_dir, f"all_{dim}.csv"), "all")
        df_ft  = _load_pca_csv(os.path.join(pca_dir, f"ft_{dim}.csv"),  "ft")
        rows.append(_ft_gap(df_all, df_ft, label))

    # pca=30: from main benchmark directories, filtered to same datasets
    img30 = _load_dir(os.path.join(_RESULTS, "images"))
    txt30 = _load_dir(os.path.join(_RESULTS, "text"))
    df30  = pd.concat([img30, txt30], ignore_index=True)
    df30  = df30[df30["dataset"].isin(base_ds)]
    rows.append(_ft_gap(
        df30[df30["multimodal_state"] == "all"],
        df30[df30["multimodal_state"] == "ft"],
        "30 dims\n(default)",
    ))

    # no-PCA
    df_np = pd.read_csv(os.path.join(pca_dir, "no_pca.csv"))
    df_np["model"]      = df_np["model"].str.strip()
    df_np["test_score"] = df_np["test_score"].clip(lower=-0.1)
    rows.append(_ft_gap(
        df_np[df_np["multimodal_state"] == "all"],
        df_np[df_np["multimodal_state"] == "ft"],
        "no PCA\n(384 dims)",
    ))

    # Display order: no-PCA, 15, 30, 60
    order = ["no PCA\n(384 dims)", "15 dims", "30 dims\n(default)", "60 dims"]
    df = pd.DataFrame(rows)
    df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)
    return df.sort_values("label").reset_index(drop=True)


def _tfidf_data():
    tfidf_raw = pd.read_csv(os.path.join(_RESULTS, "analysis_tfidf/tfidf.csv"))
    tfidf_raw["dataset"]    = tfidf_raw["Name"].apply(lambda x: "_".join(x.split("_")[1:-1]))
    tfidf_raw["model"]      = tfidf_raw["model"].str.strip()
    tfidf_raw["test_score"] = tfidf_raw["test_score"].clip(lower=-0.1)
    tfidf_ds = set(tfidf_raw["dataset"].unique())

    txt = _load_dir(os.path.join(_RESULTS, "text"))
    txt = txt[txt["dataset"].isin(tfidf_ds) & txt["multimodal_state"].isin(["all", "ft"])]

    tfidf_raw["condition"] = "TF-IDF\n(frozen)"
    txt_all = txt[txt["multimodal_state"] == "all"].copy()
    txt_ft  = txt[txt["multimodal_state"] == "ft"].copy()
    txt_all["condition"] = "E5-small\n(frozen)"
    txt_ft["condition"]  = "E5-small\n(fine-tuned)"

    combined = pd.concat([
        tfidf_raw[["dataset", "fold", "model", "condition", "test_score"]],
        txt_all[["dataset",   "fold", "model", "condition", "test_score"]],
        txt_ft[["dataset",    "fold", "model", "condition", "test_score"]],
    ], ignore_index=True)
    norm = _normalize_within(combined)
    return _agg_norm(norm, "condition")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, title, xlabel):
    ax.set_title(title, fontsize=13, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.tick_params(axis="both", labelsize=10)


def _hbar_gap(ax, df, color, sep_after=None):
    y = np.arange(len(df))
    for i, row in df.iterrows():
        ax.barh(y[i], row["mean"], 0.45, color=color,
                edgecolor="white", linewidth=0.8, zorder=2)
        ax.errorbar(row["mean"], y[i], xerr=row["ci"],
                    fmt="none", ecolor="#444", capsize=2.5, linewidth=0.8, zorder=3)
    if sep_after is not None:
        ax.axhline(y=sep_after + 0.5, color="#888", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=10)
    ax.axvline(0, color="#888", linewidth=0.7, zorder=1)


def _hbar_norm(ax, df, order, colors):
    label_to_color = dict(zip(order, colors))
    y = np.arange(len(order))
    for i, lbl in enumerate(order):
        row = df[df["condition"] == lbl].iloc[0]
        c = label_to_color[lbl]
        ax.barh(y[i], row["mean"], 0.45, color=c,
                edgecolor="white", linewidth=0.8, zorder=2)
        ax.errorbar(row["mean"], y[i], xerr=row["ci"],
                    fmt="none", ecolor="#444", capsize=2.5, linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=10)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure():
    enc_df   = _encoder_data()
    pca_df   = _pca_data()
    tfidf_df = _tfidf_data()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    fig.subplots_adjust(wspace=0.55, left=0.10, right=0.97, top=0.88, bottom=0.12)

    # ── Panel (a): Encoder scale ────────────────────────────────────────────
    ax = axes[0]
    enc_df = enc_df.reset_index(drop=True)
    _hbar_gap(ax, enc_df, color=_COLOR_FINETUNED, sep_after=1)
    vals = enc_df["mean"]
    margin = max(enc_df["ci"].max() * 1.5, 0.003)
    ax.set_xlim(0, vals.max() + margin)
    _style_ax(ax, "(a) Encoder scale", "FT improvement (ft \u2212 frozen)")
    ax.annotate("image", xy=(0.02, 0.83), xycoords="axes fraction",
                fontsize=7.5, color="#555", style="italic")
    ax.annotate("text",  xy=(0.02, 0.33), xycoords="axes fraction",
                fontsize=7.5, color="#555", style="italic")

    # ── Panel (b): PCA dimensions ───────────────────────────────────────────
    ax = axes[1]
    pca_df = pca_df.reset_index(drop=True)
    _hbar_gap(ax, pca_df, color=_COLOR_FINETUNED)
    vals = pca_df["mean"]
    margin = max(pca_df["ci"].max() * 1.5, 0.003)
    ax.set_xlim(0, vals.max() + margin)
    _style_ax(ax, "(b) PCA dimensions", "FT improvement (ft \u2212 frozen)")

    # ── Panel (c): TF-IDF vs E5 ─────────────────────────────────────────────
    ax = axes[2]
    tf_order  = ["TF-IDF\n(frozen)", "E5-small\n(frozen)", "E5-small\n(fine-tuned)"]
    tf_colors = [_COLOR_TFIDF, _COLOR_FROZEN, _COLOR_FINETUNED]
    _hbar_norm(ax, tfidf_df, tf_order, tf_colors)
    norm_vals = tfidf_df["mean"]
    margin = max(tfidf_df["ci"].max() * 1.5, 0.03)
    ax.set_xlim(max(0, norm_vals.min() - margin), min(1, norm_vals.max() + margin))
    _style_ax(ax, "(c) Text representation", "Normalized score (per-dataset min/max)")
    legend_tf = [
        Patch(color=_COLOR_TFIDF,     label="TF-IDF"),
        Patch(color=_COLOR_FROZEN,    label="E5 frozen"),
        Patch(color=_COLOR_FINETUNED, label="E5 fine-tuned"),
    ]
    ax.legend(handles=legend_tf, fontsize=10, frameon=True,
              framealpha=0.85, edgecolor="none", loc="lower right")

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    fig.savefig(_OUT, format="pdf", dpi=200, bbox_inches="tight")
    print(f"Saved to {_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
