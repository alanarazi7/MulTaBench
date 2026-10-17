"""Writes the CSVs the leaderboard Space shows, computed from the per-model run CSVs."""
import argparse
import itertools
import os

import pandas as pd

from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.datasets.info import Tier, dataset_info
from multabench.leaderboard.elo import RESULTS_DIR, error_table, leaderboard, load_runs
from multabench.leaderboard.embedding_cost import EMBEDDING_COST_CSV
from multabench.result_keys import ResultKey, RunStatus

SPACE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "space", "data")
TIERS = [t.value for t in Tier]
TASKS = ["classification", "regression"]
MODALITIES = [m.value for m in Modality]


def dataset_tier(dataset: str) -> str:
    return dataset_info(dataset).tier.value


def dataset_task(dataset: str) -> str:
    return "regression" if dataset.startswith("REG_") else "classification"


def _subset_key(picked: list[str], values: list[str]) -> str:
    return "all" if len(picked) == len(values) else "+".join(picked)


def subsets(values: list[str]) -> list[list[str]]:
    return [list(c) for n in range(1, len(values) + 1) for c in itertools.combinations(values, n)]


def leaderboards(runs: pd.DataFrame, bootstraps: int = 1000) -> pd.DataFrame:
    """One Elo leaderboard per combination of tiers, tasks and modalities, keyed by `tier`, `task` and `modality`."""
    datasets = runs[ResultKey.DATASET]
    tier, task = datasets.map(dataset_tier), datasets.map(dataset_task)
    modality = datasets.map(lambda d: dataset_modality(MulTaBenchDatasetID[d]).value)
    boards = []
    for tiers, tasks, modalities in itertools.product(subsets(TIERS), subsets(TASKS), subsets(MODALITIES)):
        keep = tier.isin(tiers) & task.isin(tasks) & modality.isin(modalities)
        board = leaderboard(error_table(runs[keep]), bootstraps=bootstraps)
        board.insert(0, "modality", _subset_key(modalities, MODALITIES))
        board.insert(0, "task", _subset_key(tasks, TASKS))
        board.insert(0, "tier", _subset_key(tiers, TIERS))
        boards.append(board)
    return pd.concat(boards, ignore_index=True)


def per_dataset(runs: pd.DataFrame) -> pd.DataFrame:
    ok = runs[runs[ResultKey.STATUS] == RunStatus.OK]
    df = ok.groupby([ResultKey.DATASET, ResultKey.MODEL]).agg(**{
        ResultKey.METRIC: (ResultKey.METRIC, "first"), ResultKey.TEST_ERROR: (ResultKey.TEST_ERROR, "mean"),
        "test_error_std": (ResultKey.TEST_ERROR, "std"), ResultKey.TEST_SCORE: (ResultKey.TEST_SCORE, "mean"),
        "folds": (ResultKey.FOLD, "nunique"), ResultKey.N_TRAIN: (ResultKey.N_TRAIN, "max")}).reset_index()
    modality = df[ResultKey.DATASET].map(lambda d: dataset_modality(MulTaBenchDatasetID[d]).value)
    df.insert(1, "modality", modality)
    df.insert(1, "task", df[ResultKey.DATASET].map(dataset_task))
    df.insert(1, "tier", df[ResultKey.DATASET].map(dataset_tier))
    return df


def cost(runs: pd.DataFrame, embedding_cost_csv: str = EMBEDDING_COST_CSV) -> pd.DataFrame:
    """Median train and predict time per 1K rows, which leave out reading the cached embeddings, and the mean time the
    run's frozen encoders would take to embed 1K rows live, an estimate from the caches' own encoding time."""
    ok = runs[runs[ResultKey.STATUS] == RunStatus.OK]
    encode = pd.read_csv(embedding_cost_csv).set_index(["dataset", "encoder"])["encode_s_per_1k"]

    def encode_s(col: str) -> pd.Series:
        keys = list(zip(ok[ResultKey.DATASET], ok[col]))
        return pd.Series([encode.get(k, 0.0) for k in keys], index=ok.index)

    ok = ok.assign(text_encode=encode_s(ResultKey.TEXT_ENCODER), image_encode=encode_s(ResultKey.IMAGE_ENCODER))
    df = ok.groupby(ResultKey.MODEL).agg(
        train_s_per_1k=(ResultKey.TRAIN_TIME_PER_1K, "median"),
        inference_s_per_1k=(ResultKey.INFERENCE_TIME_PER_1K, "median"),
        text_encode_s_per_1k=("text_encode", "mean"), image_encode_s_per_1k=("image_encode", "mean"),
        gpu=(ResultKey.GPU_TYPE, "first"), cpu=(ResultKey.CPU_NAME, "first")).reset_index()
    return df.sort_values("train_s_per_1k", ignore_index=True)


def export(out_dir: str, results_dir: str = RESULTS_DIR, bootstraps: int = 1000):
    os.makedirs(out_dir, exist_ok=True)
    runs = load_runs(results_dir)
    leaderboards(runs, bootstraps).round(4).to_csv(os.path.join(out_dir, "leaderboards.csv"), index=False)
    per_dataset(runs).round(6).to_csv(os.path.join(out_dir, "per_dataset.csv"), index=False)
    cost(runs).round(4).to_csv(os.path.join(out_dir, "cost.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the CSVs of the leaderboard Space.")
    parser.add_argument('--out_dir', type=str, default=SPACE_DATA_DIR)
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR)
    args = parser.parse_args()
    export(args.out_dir, args.results_dir)
