"""Evaluation metrics, as in TabArena: ROC AUC (binary), log loss (multiclass) and RMSE (regression).

The same computation as AutoGluon's scorers, which TabArena uses, in scikit-learn alone.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from multabench.datasets.objects import SupervisedTask

METRICS = {SupervisedTask.BINARY: "roc_auc", SupervisedTask.MULTICLASS: "log_loss",
           SupervisedTask.REGRESSION: "root_mean_squared_error"}


@dataclass(frozen=True)
class Score:
    """test_score is higher-is-better (AUC, or minus the loss); test_error is TabArena's metric_error, 0 perfect."""
    metric: str
    test_score: float
    test_error: float


def score(task_type: SupervisedTask, y_true, y_pred) -> Score:
    """y_true holds the label-encoded classes 0..K-1 for classification; y_pred the positive-class probability for
    binary (or both columns), one probability column per class for multiclass, and the prediction for regression."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if task_type == SupervisedTask.BINARY:
        if y_pred.ndim == 2:
            y_pred = y_pred[:, 1]
        auc = float(roc_auc_score(y_true, y_pred))
        return Score(metric=METRICS[task_type], test_score=auc, test_error=1 - auc)
    if task_type == SupervisedTask.MULTICLASS:
        loss = float(log_loss(y_true.astype(np.int32), y_pred, labels=np.arange(y_pred.shape[1], dtype=np.int32)))
    else:
        loss = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    return Score(metric=METRICS[task_type], test_score=-loss, test_error=loss)
