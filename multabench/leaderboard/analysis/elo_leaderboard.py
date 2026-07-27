"""Bootstrapped Bradley-Terry / Elo leaderboard for MulTaBench default results.

A faithful reimplementation of the LMSYS Chatbot-Arena / TabArena method (log-odds MLE via
logistic regression, 400-point Elo scale base-10, anchored so a reference model = 1000, CIs by
bootstrapping over datasets). NOT TabArena's code verbatim: its `compare()` consumes a cached
tabrepo results object keyed on error metrics, not our per-(dataset, model, fold) score CSVs.

Design choices (all easy to revise):
- Each model is represented by its DEFAULT config: TAR ("ft") where it exists, else "all"
  (AutoGluon-MM has no TAR variant, so it uses its single end-to-end score).
- A "battle" is per dataset, on the mean score over the 5 folds. Model A beats B on a dataset
  if its mean score is higher. AUC (classification) and R^2 (regression) are both
  higher-is-better and are only ever compared WITHIN a dataset, so mixing the two metrics is
  fine -- the comparison is purely rank-based and scale-free.
- Missing (model, dataset) cells (TabPFNv2/TabPFN-2.5 on the 2 high-multiclass datasets they
  can't run) simply produce no battle -- Bradley-Terry handles this natively, no imputation.
- Ratings are anchored so RandomForest = 1000 (mirroring TabArena's default-RF anchor); the
  anchor is re-applied inside every bootstrap replicate, so CIs are relative to RandomForest.

Two leaderboards are produced:
- TAR-only (elo_leaderboard.csv): each of the 13 models once, at its default config -- the
  clean "which model is best" ranking.
- Frozen-vs-TAR variants (elo_frozen_vs_tar.csv): the 10 embedding models each split into a
  "(Frozen)" and a "(TAR)" competitor, plus the 3 E2E models as single entries (23 total),
  anchored to RandomForest (Frozen) = 1000. This expresses the paper's core claim in Elo
  terms -- every embedding model's TAR variant outranks its own Frozen variant -- and reports
  the per-model TAR uplift in Elo points.

Run standalone: `python -m multabench.leaderboard.analysis.elo_leaderboard`
"""
import glob
import itertools
import math
from os.path import dirname, join

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_RES = join(dirname(__file__), "..", "results")
_OUT_CSV = join(_RES, "analysis_curation_sensitivity", "elo_leaderboard.csv")
_OUT_VARIANT_CSV = join(_RES, "analysis_curation_sensitivity", "elo_frozen_vs_tar.csv")

MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost", "TabM Ⓜ️": "TabM",
    "TabPFN-v2 🤯": "TabPFNv2", "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
    "TabDPT 6️⃣": "TabDPT", "TabICLv2 🗼": "TabICLv2", "RealMLP 🕸": "RealMLP",
    "XGBoost 🌲": "XGBoost", "RandomForest 🌳": "RandomForest",
    "AutoGluon-MM 🧴": "AutoGluon-MM", "ConTextTab 🏢": "ConTextTab", "TabSTAR ⭐": "TabSTAR",
}
ANCHOR = "RandomForest"
SCALE, BASE, INIT = 400, 10, 1000

# For the Frozen-vs-TAR variant leaderboard: the 10 models that consume DINO/E5 embeddings
# (so "all"=Frozen and "ft"=TAR is exactly the paper's frozen-vs-target-aware comparison), and
# the 3 end-to-end models, which natively process the modality and are split by modality+state.
EMBEDDING_MODELS = ["LightGBM", "CatBoost", "TabM", "TabPFNv2", "TabPFN-2.5",
                    "TabDPT", "TabICLv2", "RealMLP", "XGBoost", "RandomForest"]
E2E_MODELS = ["AutoGluon-MM", "TabSTAR", "ConTextTab"]
# Modality-split E2E models (TabSTAR and ConTextTab split by image/text and frozen/tar)
E2E_SPLIT_MODELS = ["TabSTAR", "ConTextTab"]
VARIANT_ANCHOR = "RandomForest (Frozen)"


def _load(files) -> pd.DataFrame:
    frames = []
    for f in files:
        d = pd.read_csv(f).dropna(subset=["model"])
        d["model"] = d["model"].str.strip().map(MODEL_LABELS)
        d = d[d["model"].notna()]
        frames.append(d[["model", "dataset", "multimodal_state", "fold", "test_score"]])
    return pd.concat(frames, ignore_index=True)


def load_default_scores() -> pd.DataFrame:
    """[dataset, model, mean_score, modality] -- each model's default-config mean over folds,
    for the 40 MulTaBench datasets (20 image from images/, 20 text from text/, plus the
    supplementary + E2E models from more_baselines/)."""
    img = _load(glob.glob(join(_RES, "images", "*.csv")))
    txt = _load(glob.glob(join(_RES, "text", "*.csv")))
    mb = _load(glob.glob(join(_RES, "more_baselines", "*.csv")))
    img_ds = set(img["dataset"].unique())
    bench_ds = img_ds | set(txt["dataset"].unique())

    allrows = pd.concat([img, txt, mb], ignore_index=True)
    allrows = allrows[allrows["multimodal_state"].isin(["all", "ft"])]
    allrows = allrows[allrows["dataset"].isin(bench_ds)]

    rows = []
    for (model, dataset), g in allrows.groupby(["model", "dataset"]):
        state = "ft" if "ft" in set(g["multimodal_state"]) else "all"
        rows.append({"dataset": dataset, "model": model,
                     "mean_score": g[g["multimodal_state"] == state]["test_score"].mean()})
    out = pd.DataFrame(rows)
    out["modality"] = np.where(out["dataset"].isin(img_ds), "image", "text")
    return out


def load_variant_scores() -> pd.DataFrame:
    """[dataset, model, mean_score, modality] where `model` is a COMPETITOR label: each of the
    10 embedding models split into "<name> (Frozen)" (all) and "<name> (TAR)" (ft), plus the
    modality+state variants of split E2E models (TabSTAR, ConTextTab) and AutoGluon-MM as-is."""
    img = _load(glob.glob(join(_RES, "images", "*.csv")))
    txt = _load(glob.glob(join(_RES, "text", "*.csv")))
    mb = _load(glob.glob(join(_RES, "more_baselines", "*.csv")))
    img_ds = set(img["dataset"].unique())
    bench_ds = img_ds | set(txt["dataset"].unique())

    allrows = pd.concat([img, txt, mb], ignore_index=True)
    allrows = allrows[allrows["multimodal_state"].isin(["all", "ft"])]
    allrows = allrows[allrows["dataset"].isin(bench_ds)]

    rows = []
    for (model, dataset), g in allrows.groupby(["model", "dataset"]):
        states = set(g["multimodal_state"])
        modality = "image" if dataset in img_ds else "text"

        if model in EMBEDDING_MODELS:
            # Embedding models: split into Frozen and TAR variants
            if "all" in states:
                rows.append({"dataset": dataset, "model": f"{model} (Frozen)",
                             "mean_score": g[g["multimodal_state"] == "all"]["test_score"].mean()})
            if "ft" in states:
                rows.append({"dataset": dataset, "model": f"{model} (TAR)",
                             "mean_score": g[g["multimodal_state"] == "ft"]["test_score"].mean()})
        elif model in E2E_SPLIT_MODELS:
            # Split E2E models (TabSTAR, ConTextTab): split by modality and state
            if "all" in states:
                rows.append({"dataset": dataset, "model": f"{model}-{modality.title()}-Frozen",
                             "mean_score": g[g["multimodal_state"] == "all"]["test_score"].mean()})
            if "ft" in states:
                rows.append({"dataset": dataset, "model": f"{model}-{modality.title()}-TAR",
                             "mean_score": g[g["multimodal_state"] == "ft"]["test_score"].mean()})
        else:
            # Other E2E models (AutoGluon-MM): single competitor at default config (TAR-preferred)
            state = "ft" if "ft" in states else "all"
            rows.append({"dataset": dataset, "model": f"{model}",
                         "mean_score": g[g["multimodal_state"] == state]["test_score"].mean()})
    out = pd.DataFrame(rows)
    out["modality"] = np.where(out["dataset"].isin(img_ds), "image", "text")
    return out


def _battles(wide: pd.DataFrame, models):
    """One row per (dataset, model-pair): winner-perspective design vector + weight (ties get
    half weight each way)."""
    idx = {m: i for i, m in enumerate(models)}
    X, Y, W = [], [], []
    lb = math.log(BASE)
    for _, row in wide.iterrows():
        present = [m for m in models if pd.notna(row[m])]
        for a, b in itertools.combinations(present, 2):
            v = np.zeros(len(models))
            v[idx[a]], v[idx[b]] = lb, -lb
            sa, sb = row[a], row[b]
            if sa > sb:
                X.append(v); Y.append(1); W.append(1.0)
            elif sa < sb:
                X.append(v); Y.append(0); W.append(1.0)
            else:
                X.append(v); Y.append(1); W.append(0.5)
                X.append(v.copy()); Y.append(0); W.append(0.5)
    return np.array(X), np.array(Y), np.array(W)


def fit_elo(wide: pd.DataFrame, models, anchor: str = ANCHOR) -> pd.Series:
    X, Y, W = _battles(wide, models)
    if len(X) == 0 or len(np.unique(Y)) < 2:
        return pd.Series(float(INIT), index=models)
    lr = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
    lr.fit(X, Y, sample_weight=W)
    elo = SCALE * lr.coef_[0] + INIT
    elo = elo - elo[models.index(anchor)] + INIT
    return pd.Series(elo, index=models)


def leaderboard(scores: pd.DataFrame, order, anchor: str = ANCHOR,
                n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    wide = scores.pivot(index="dataset", columns="model", values="mean_score")
    models = [m for m in order if m in wide.columns]
    point = fit_elo(wide, models, anchor)

    rng = np.random.default_rng(seed)
    datasets = wide.index.to_numpy()
    boot = np.zeros((n_boot, len(models)))
    for i in range(n_boot):
        samp = rng.choice(datasets, size=len(datasets), replace=True)
        boot[i] = fit_elo(wide.loc[samp].reset_index(drop=True), models, anchor).values
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    lb = pd.DataFrame({
        "model": models,
        "elo": point.values.round(0).astype(int),
        "ci_low": lo.round(0).astype(int),
        "ci_high": hi.round(0).astype(int),
        "n_datasets": [int(wide[m].notna().sum()) for m in models],
    }).sort_values("elo", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", range(1, len(lb) + 1))
    return lb


def _print(lb):
    show = lb.copy()
    show["ci"] = "+" + (show.ci_high - show.elo).astype(str) + " / -" + (show.elo - show.ci_low).astype(str)
    print(show[["rank", "model", "elo", "ci", "n_datasets"]].to_string(index=False))


def tar_uplift(variant_lb: pd.DataFrame) -> pd.DataFrame:
    """Per model, Elo(TAR) - Elo(Frozen) -- the headline "how much does TAR help"."""
    e = variant_lb.set_index("model")["elo"]
    rows = []

    # Embedding models
    for m in EMBEDDING_MODELS:
        f, t = f"{m} (Frozen)", f"{m} (TAR)"
        if f in e.index and t in e.index:
            rows.append({"model": m, "type": "embedding", "elo_frozen": int(e[f]), "elo_tar": int(e[t]),
                         "tar_uplift": int(e[t] - e[f])})

    # Split E2E models (by modality)
    for m in E2E_SPLIT_MODELS:
        for mod in ["Image", "Text"]:
            f, t = f"{m}-{mod}-Frozen", f"{m}-{mod}-TAR"
            if f in e.index and t in e.index:
                rows.append({"model": f"{m} ({mod})", "type": "e2e", "elo_frozen": int(e[f]), "elo_tar": int(e[t]),
                             "tar_uplift": int(e[t] - e[f])})

    return pd.DataFrame(rows).sort_values("tar_uplift", ascending=False).reset_index(drop=True)


def main():
    order = list(MODEL_LABELS.values())
    scores = load_default_scores()
    print(f"Loaded {scores['dataset'].nunique()} datasets, {scores['model'].nunique()} models")

    frames = []
    for label, sub in [("combined", scores),
                       ("image", scores[scores.modality == "image"]),
                       ("text", scores[scores.modality == "text"])]:
        lb = leaderboard(sub, order=order)
        lb.insert(0, "split", label)
        frames.append(lb)
        print(f"\n=== TAR-only, {label.upper()} ({sub['dataset'].nunique()} datasets) ===")
        _print(lb)
    pd.concat(frames, ignore_index=True).to_csv(_OUT_CSV, index=False)
    print(f"\nWrote {_OUT_CSV}")

    # --- Frozen-vs-TAR variant leaderboard ---
    variant = load_variant_scores()
    # Order: embeddings Frozen, embeddings TAR, split E2E (Image Frozen, Image TAR, Text Frozen, Text TAR), AutoGLUON
    variant_order = ([f"{m} (Frozen)" for m in EMBEDDING_MODELS]
                     + [f"{m} (TAR)" for m in EMBEDDING_MODELS]
                     + [f"{m}-Image-Frozen" for m in E2E_SPLIT_MODELS]
                     + [f"{m}-Image-TAR" for m in E2E_SPLIT_MODELS]
                     + [f"{m}-Text-Frozen" for m in E2E_SPLIT_MODELS]
                     + [f"{m}-Text-TAR" for m in E2E_SPLIT_MODELS]
                     + ["AutoGluon-MM"])
    vframes = []
    for label, sub in [("combined", variant),
                       ("image", variant[variant.modality == "image"]),
                       ("text", variant[variant.modality == "text"])]:
        lb = leaderboard(sub, order=variant_order, anchor=VARIANT_ANCHOR)
        lb.insert(0, "split", label)
        vframes.append(lb)
        print(f"\n=== Frozen-vs-TAR variants, {label.upper()} ({sub['dataset'].nunique()} datasets) ===")
        _print(lb)
        if label == "combined":
            up = tar_uplift(lb)
            print("\n  TAR uplift (Elo TAR - Elo Frozen), per model:")
            print(up.to_string(index=False))
            embedding_only = up[up["type"] == "embedding"]
            print(f"\n  All {len(embedding_only)} embedding models: TAR > Frozen? {(embedding_only.tar_uplift > 0).all()}")
            print(f"  Median TAR uplift (embeddings): {embedding_only.tar_uplift.median():.0f} Elo   "
                  f"(mean {embedding_only.tar_uplift.mean():.0f})")
            e2e_only = up[up["type"] == "e2e"]
            if not e2e_only.empty:
                print(f"\n  E2E models TAR uplift:")
                print(e2e_only.to_string(index=False))
    pd.concat(vframes, ignore_index=True).to_csv(_OUT_VARIANT_CSV, index=False)
    print(f"\nWrote {_OUT_VARIANT_CSV}")


if __name__ == "__main__":
    main()
