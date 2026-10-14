import glob
import json
import os
from typing import List

import pandas as pd

from multabench.result_keys import RUN_KEY, ResultKey

RESULT_COLUMNS = [
    *RUN_KEY, ResultKey.STATUS, ResultKey.METRIC, ResultKey.TEST_SCORE, ResultKey.TEST_ERROR,
    ResultKey.TRAIN_TIME_PER_1K, ResultKey.INFERENCE_TIME_PER_1K, ResultKey.TRAIN_EMBEDDING_TIME_PER_1K,
    ResultKey.INFERENCE_EMBEDDING_TIME_PER_1K, ResultKey.N_TRAIN, ResultKey.N_TEST, ResultKey.GPU_TYPE,
    ResultKey.CPU_NAME, ResultKey.CPU_CORES, ResultKey.GIT, ResultKey.TIMESTAMP, ResultKey.ERROR,
]


def collect_results(output_dir: str) -> pd.DataFrame:
    """One row per run JSON in output_dir, failed runs included; keys a run didn't write are left empty."""
    rows = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)


def write_model_csvs(df: pd.DataFrame, results_dir: str) -> List[str]:
    """Writes one CSV per model; a model's earlier rows are kept unless a new run has the same key."""
    os.makedirs(results_dir, exist_ok=True)
    paths = []
    for model, runs in df.groupby(ResultKey.MODEL):
        path = os.path.join(results_dir, f"{model}.csv")
        if os.path.exists(path):
            runs = pd.concat([pd.read_csv(path), runs]).drop_duplicates(RUN_KEY, keep="last")
        runs.sort_values(RUN_KEY).to_csv(path, index=False)
        paths.append(path)
    return paths
