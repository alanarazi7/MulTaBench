"""Evaluation metrics, as in TabArena: ROC AUC (binary), log loss (multiclass) and RMSE (regression).

Scored with AutoGluon's scorers, the same implementation TabArena uses.
"""
from dataclasses import dataclass
from typing import Union

import numpy as np
from autogluon.core.metrics import Scorer, log_loss, roc_auc, root_mean_squared_error
from pandas import Series


@dataclass
class Metrics:
    metric: str
    # Higher is better: the metric itself for AUC, its negation for RMSE and log loss.
    score: float
    # Lower is better, 0 is perfect: TabArena's metric_error.
    error: float


def get_scorer(d_output: int) -> Scorer:
    if d_output == 1:
        return root_mean_squared_error
    if d_output == 2:
        return roc_auc
    if d_output > 2:
        return log_loss
    raise ValueError(f"Unsupported d_output: {d_output}. Expected 1 (regression), 2 (binary), or >2 (multiclass).")


def calculate_metric(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, d_output: int) -> Metrics:
    """y_true holds the label-encoded classes (0..d_output-1) for classification; y_pred holds the
    positive-class probability for binary, one probability column per class for multiclass."""
    scorer = get_scorer(d_output)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if d_output == 2 and y_pred.ndim == 2:
        y_pred = y_pred[:, 1]
    score = float(scorer(y_true, y_pred))
    return Metrics(metric=scorer.name, score=score, error=float(scorer.convert_score_to_error(score)))
