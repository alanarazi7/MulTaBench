"""Elo leaderboard over the per-model run CSVs, computed by TabArena's bencheval.

One battle per dataset fold and model pair, on that fold's test_error, fit as a Bradley-Terry model with RandomForest
at 1000, and 95% intervals from bootstrapping the datasets, each drawn with all its folds. As in TabArena, the intervals
are bootstrapped before anchoring, so each is the model's own, apart from RandomForest's. Win rate, average rank,
normalized error and best-on count are per fold too. A model with no successful run on a fold has no battles there:
nothing is imputed.
"""
import argparse
import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.result_keys import ResultKey, RunStatus

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "models")
# Runs with fine-tuned (-tar) encoders, with frozen large ones, and with fine-tuned large ones.
TAR_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "models_tar")
LARGE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "models_large")
LARGE_TAR_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "models_large_tar")
ANCHOR = "RandomForest"
INIT = 1000


def load_runs(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path) for path in sorted(glob.glob(os.path.join(results_dir, "*.csv")))])


def error_table(runs: pd.DataFrame, modalities: Optional[List[Modality]] = None) -> pd.DataFrame:
    """The successful runs' test_error, one row per dataset and fold and one column per model."""
    ok = runs[runs[ResultKey.STATUS] == RunStatus.OK]
    if modalities:
        ok = ok[ok[ResultKey.DATASET].map(lambda d: dataset_modality(MulTaBenchDatasetID[d]) in modalities)]
    return ok.groupby([ResultKey.DATASET, ResultKey.FOLD, ResultKey.MODEL])[ResultKey.TEST_ERROR].mean().unstack()


def _datasets(table: pd.DataFrame) -> pd.Index:
    """The dataset of each row: the first index level of a per-fold table."""
    return table.index.get_level_values(0)


def elo_ratings(table: pd.DataFrame, bootstraps: int = 1000, anchor: str = ANCHOR) -> pd.DataFrame:
    """bencheval's Elo, with the 95% interval bounds, indexed by model, with the anchor at 1000."""
    from bencheval.evaluator import BenchmarkEvaluator
    named = table.rename_axis(index=[ResultKey.DATASET, ResultKey.FOLD], columns=ResultKey.MODEL)
    runs = named.stack().rename(ResultKey.TEST_ERROR).reset_index()
    evaluator = BenchmarkEvaluator(method_col=ResultKey.MODEL, task_col=ResultKey.DATASET,
                                   error_col=ResultKey.TEST_ERROR, seed_column=ResultKey.FOLD)
    elo = evaluator.compute_elo(results_per_task=runs, calibration_framework=anchor, calibration_elo=INIT,
                                BOOTSTRAP_ROUNDS=bootstraps, round_decimals=None)
    return pd.DataFrame({"elo": elo["elo"], "elo_low": elo["elo"] - elo["elo-"], "elo_high": elo["elo"] + elo["elo+"]})


def leaderboard(table: pd.DataFrame, bootstraps: int = 1000, anchor: str = ANCHOR) -> pd.DataFrame:
    models = list(table.columns)
    elo = elo_ratings(table, bootstraps=bootstraps, anchor=anchor).loc[models]
    best, worst = table.min(axis=1), table.max(axis=1)
    normalized = table.sub(best, axis=0).div((worst - best).replace(0, np.nan), axis=0)
    board = pd.DataFrame({
        ResultKey.MODEL: models, "elo": elo["elo"].values, "elo_low": elo["elo_low"].values,
        "elo_high": elo["elo_high"].values,
        "datasets": table.notna().groupby(_datasets(table)).any().sum().astype(int).values,
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
        ci = f"+{r.elo_high - r.elo:.0f} / −{r.elo - r.elo_low:.0f}"
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
