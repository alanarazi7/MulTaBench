"""A run's metric, scored by multabench.benchmark.metrics; d_output is 1 (regression), 2 (binary) or more (multiclass)."""
from typing import Dict, Union

import numpy as np
from pandas import Series

from multabench.benchmark.metrics import score
from multabench.datasets.objects import SupervisedTask
from multabench.result_keys import ResultKey


def d_output_task(d_output: int) -> SupervisedTask:
    if d_output == 1:
        return SupervisedTask.REGRESSION
    if d_output == 2:
        return SupervisedTask.BINARY
    if d_output > 2:
        return SupervisedTask.MULTICLASS
    raise ValueError(f"Unsupported d_output: {d_output}. Expected 1 (regression), 2 (binary), or >2 (multiclass).")


def calculate_metric(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, d_output: int) -> Dict:
    s = score(d_output_task(d_output), y_true=y_true, y_pred=y_pred)
    return {ResultKey.METRIC: s.metric, ResultKey.TEST_SCORE: s.test_score, ResultKey.TEST_ERROR: s.test_error}
