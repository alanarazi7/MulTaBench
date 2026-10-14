"""Elo leaderboard over the per-model run CSVs, computed as in the paper's elo_leaderboard.py.

One battle per dataset and model pair on the mean test_error over the folds, fit as a Bradley-Terry model with
RandomForest at 1000, and 95% intervals from bootstrapping the datasets. A model with no successful run on a dataset
has no battles there: nothing is imputed.
"""
import argparse
import glob
import itertools
import math
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.result_keys import STATUS, TEST_ERROR, RunStatus

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "models")
ANCHOR = "RandomForest"
SCALE, BASE, INIT = 400, 10, 1000


def load_runs(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path) for path in sorted(glob.glob(os.path.join(results_dir, "*.csv")))])


def error_table(runs: pd.DataFrame, modalities: Optional[List[Modality]] = None) -> pd.DataFrame:
    """Mean test_error over the successful folds, one row per dataset and one column per model."""
    ok = runs[runs[STATUS] == RunStatus.OK]
    if modalities:
        ok = ok[ok["dataset"].map(lambda d: dataset_modality(MulTaBenchDatasetID[d]) in modalities)]
    return ok.groupby(["dataset", "model"])[TEST_ERROR].mean().unstack()


def _battles(table: pd.DataFrame):
    """The logistic regression rows of every battle, and the dataset each row comes from."""
    models = list(table.columns)
    X, y, w, rows_dataset = [], [], [], []
    for i, (_, errors) in enumerate(table.iterrows()):
        present = [m for m in models if pd.notna(errors[m])]
        for a, b in itertools.combinations(present, 2):
            x = np.zeros(len(models))
            x[models.index(a)], x[models.index(b)] = math.log(BASE), -math.log(BASE)
            outcomes = [(1, 0.5), (0, 0.5)] if errors[a] == errors[b] else [(int(errors[a] < errors[b]), 1.0)]
            for outcome, weight in outcomes:
                X.append(x)
                y.append(outcome)
                w.append(weight)
                rows_dataset.append(i)
    return np.array(X), np.array(y), np.array(w), np.array(rows_dataset)


def _fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, models: List[str]) -> np.ndarray:
    lr = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(X, y, sample_weight=w)
    elo = SCALE * lr.coef_[0] + INIT
    return elo - elo[models.index(ANCHOR)] + INIT


def leaderboard(table: pd.DataFrame, bootstraps: int = 1000, seed: int = 0) -> pd.DataFrame:
    models = list(table.columns)
    X, y, w, rows_dataset = _battles(table)
    elo = _fit(X, y, w, models)
    rng = np.random.default_rng(seed)
    # Resampling the datasets with replacement weighs each dataset's battles by how often it was drawn.
    draws = [np.bincount(rng.choice(len(table), len(table)), minlength=len(table)) for _ in range(bootstraps)]
    boot = np.array([_fit(X, y, w * counts[rows_dataset], models) for counts in draws])
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    best, worst = table.min(axis=1), table.max(axis=1)
    normalized = table.sub(best, axis=0).div((worst - best).replace(0, np.nan), axis=0)
    board = pd.DataFrame({
        "model": models, "elo": elo, "elo_low": low, "elo_high": high,
        "datasets": table.notna().sum().values,
        "win_rate": [_win_rate(table, m) for m in models],
        "avg_rank": table.rank(axis=1).mean().values,
        "normalized_error": normalized.mean().values,
        "best_on": table.eq(best, axis=0).sum().values,
    })
    return board.sort_values("elo", ascending=False, ignore_index=True)


def _win_rate(table: pd.DataFrame, model: str) -> float:
    wins = battles = 0.0
    for other in table.columns.drop(model):
        both = table[[model, other]].dropna()
        battles += len(both)
        wins += (both[model] < both[other]).sum() + 0.5 * (both[model] == both[other]).sum()
    return wins / battles


def to_markdown(board: pd.DataFrame, n_datasets: int) -> str:
    lines = ["| # | Model | Elo | 95% CI | Datasets | Win rate | Avg rank | Normalized error | Best on |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in board.iterrows():
        ci = "anchor" if r.model == ANCHOR else f"+{r.elo_high - r.elo:.0f} / −{r.elo - r.elo_low:.0f}"
        lines.append(f"| {i + 1} | {r.model} | {r.elo:.0f} | {ci} | {r.datasets}/{n_datasets} | {r.win_rate:.1%} "
                     f"| {r.avg_rank:.2f} | {r.normalized_error:.3f} | {r.best_on} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the Elo leaderboard of the per-model run CSVs.")
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--modalities', type=str, default=None, help="e.g. text or image,image-text; default: all")
    parser.add_argument('--bootstraps', type=int, default=1000)
    args = parser.parse_args()

    modalities = [Modality(m) for m in args.modalities.split(",")] if args.modalities else None
    table = error_table(load_runs(args.results_dir), modalities)
    print(to_markdown(leaderboard(table, bootstraps=args.bootstraps), n_datasets=len(table)))
