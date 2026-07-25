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

MODEL_LABELS = {
    "LightGBM 💡": "LightGBM", "CatBoost 😸": "CatBoost", "TabM Ⓜ️": "TabM",
    "TabPFN-v2 🤯": "TabPFNv2", "TabPFN-v2p5 🇩🇪": "TabPFN-2.5",
    "TabDPT 6️⃣": "TabDPT", "TabICLv2 🗼": "TabICLv2", "RealMLP 🕸": "RealMLP",
    "XGBoost 🌲": "XGBoost", "RandomForest 🌳": "RandomForest",
    "AutoGluon-MM 🧴": "AutoGluon-MM", "ConTextTab 🏢": "ConTextTab", "TabSTAR ⭐": "TabSTAR",
}
ANCHOR = "RandomForest"
SCALE, BASE, INIT = 400, 10, 1000


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


def fit_elo(wide: pd.DataFrame, models) -> pd.Series:
    X, Y, W = _battles(wide, models)
    if len(X) == 0 or len(np.unique(Y)) < 2:
        return pd.Series(float(INIT), index=models)
    lr = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
    lr.fit(X, Y, sample_weight=W)
    elo = SCALE * lr.coef_[0] + INIT
    elo = elo - elo[models.index(ANCHOR)] + INIT
    return pd.Series(elo, index=models)


def leaderboard(scores: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    wide = scores.pivot(index="dataset", columns="model", values="mean_score")
    models = [m for m in MODEL_LABELS.values() if m in wide.columns]
    point = fit_elo(wide, models)

    rng = np.random.default_rng(seed)
    datasets = wide.index.to_numpy()
    boot = np.zeros((n_boot, len(models)))
    for i in range(n_boot):
        samp = rng.choice(datasets, size=len(datasets), replace=True)
        boot[i] = fit_elo(wide.loc[samp].reset_index(drop=True), models).values
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


def main():
    scores = load_default_scores()
    print(f"Loaded {scores['dataset'].nunique()} datasets, {scores['model'].nunique()} models")

    frames = []
    for label, sub in [("combined", scores),
                       ("image", scores[scores.modality == "image"]),
                       ("text", scores[scores.modality == "text"])]:
        lb = leaderboard(sub)
        lb.insert(0, "split", label)
        frames.append(lb)
        print(f"\n=== {label.upper()} ({sub['dataset'].nunique()} datasets) ===")
        show = lb.copy()
        show["ci"] = "+" + (show.ci_high - show.elo).astype(str) + " / -" + (show.elo - show.ci_low).astype(str)
        print(show[["rank", "model", "elo", "ci", "n_datasets"]].to_string(index=False))

    pd.concat(frames, ignore_index=True).to_csv(_OUT_CSV, index=False)
    print(f"\nWrote {_OUT_CSV}")


if __name__ == "__main__":
    main()
