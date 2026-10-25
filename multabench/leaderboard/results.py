"""Compiles every run on the leaderboard into one table, one row per model, encoding, dataset and fold."""
import argparse

import pandas as pd

from multabench.leaderboard.export import ENCODING, board_modality, dataset_task, dataset_tier, load_entries
from multabench.result_keys import ResultKey, RunStatus

COLUMNS = [ResultKey.MODEL, ENCODING, ResultKey.DATASET, ResultKey.FOLD, ResultKey.TEXT_ENCODER,
           ResultKey.IMAGE_ENCODER, ResultKey.METRIC, ResultKey.TEST_SCORE, ResultKey.TEST_ERROR,
           ResultKey.TRAIN_TIME_PER_1K, ResultKey.INFERENCE_TIME_PER_1K, ResultKey.N_TRAIN, ResultKey.N_TEST]


def all_results() -> pd.DataFrame:
    """Every successful run, with its dataset's tier, task and modality as the leaderboard filters them."""
    runs = load_entries()
    df = runs.loc[runs[ResultKey.STATUS] == RunStatus.OK, COLUMNS]
    df = df.sort_values([ResultKey.MODEL, ENCODING, ResultKey.DATASET, ResultKey.FOLD], ignore_index=True)
    df.insert(3, "modality", df[ResultKey.DATASET].map(board_modality))
    df.insert(3, "task", df[ResultKey.DATASET].map(dataset_task))
    df.insert(3, "tier", df[ResultKey.DATASET].map(dataset_tier))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write every run on the leaderboard to one CSV.")
    parser.add_argument('out', nargs='?', default="multabench_results.csv")
    args = parser.parse_args()
    all_results().to_csv(args.out, index=False)
