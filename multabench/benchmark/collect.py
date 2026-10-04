import glob
import json
import os

import pandas as pd

from multabench.result_keys import (INFERENCE_EMBEDDING_TIME_PER_1K, INFERENCE_TIME_PER_1K, METRIC, STATUS,
                                    TEST_ERROR, TEST_SCORE, TRAIN_EMBEDDING_TIME_PER_1K, TRAIN_TIME_PER_1K)

RESULT_COLUMNS = [
    "model", "dataset", "text_encoder", "image_encoder", "size", "fold", STATUS,
    METRIC, TEST_SCORE, TEST_ERROR,
    TRAIN_TIME_PER_1K, INFERENCE_TIME_PER_1K, TRAIN_EMBEDDING_TIME_PER_1K, INFERENCE_EMBEDDING_TIME_PER_1K,
    "n_train", "n_test", "gpu_type", "cpu_name", "cpu_cores", "git", "timestamp", "error",
]


def collect_results(output_dir: str) -> pd.DataFrame:
    """One row per run JSON in output_dir, failed runs included; keys a run didn't write are left empty."""
    rows = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
