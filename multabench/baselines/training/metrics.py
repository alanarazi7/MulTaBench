from dataclasses import dataclass
from typing import Union, Dict

import numpy as np
from numpy.exceptions import AxisError
from pandas import Series
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, log_loss


@dataclass
class Metrics:
    score: float
    metrics: Dict[str, float]

    def __post_init__(self):
        self.score = float(self.score)
        self.metrics = {k: float(v) for k, v in self.metrics.items()}


def calculate_metric(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, d_output: int) -> Metrics:
    if d_output == 1:
        return _calculate_metrics_for_regression(y_true=y_true, y_pred=y_pred)
    elif d_output == 2:
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        return _calculate_metrics_for_binary(y_true=y_true, y_pred=y_pred)
    elif d_output > 2:
        return _calculate_metrics_for_multiclass(y_true=y_true, y_pred=y_pred)
    raise ValueError(f"Unsupported d_output: {d_output}. Expected 1 (regression), 2 (binary), or >2 (multiclass).")

def _calculate_metrics_for_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    rsq = r2_score(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    rmse = np.sqrt(mse)
    metrics = {'r2': rsq, 'mse': mse, 'rmse': rmse}
    return Metrics(score=rsq, metrics=metrics)

def _calculate_metrics_for_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    auc = roc_auc_score(y_true, y_pred)
    metrics = {'auc': auc}
    return Metrics(score=auc, metrics=metrics)

def _calculate_metrics_for_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    try:
        auc = roc_auc_score(y_true=y_true, y_score=y_pred, multi_class='ovr', average='macro')
    except (ValueError, AxisError):
        # Error calculating AUC, likely due to class imbalance or insufficient samples
        auc = _per_class_auc(y_true=y_true, y_pred=y_pred)
    try:
        logloss = log_loss(y_true, y_pred)
    except Exception:
        logloss = np.nan
    metrics = {'auc': auc, 'logloss': logloss}
    return Metrics(score=auc, metrics=metrics)

def _per_class_auc(y_true, y_pred) -> float:
    present_classes = np.unique(y_true)
    aucs = {}
    for cls in present_classes:
        # Binary ground truth: 1 for the current class, 0 for others
        y_true_binary = (y_true == cls).astype(int)
        # Predicted probabilities for the current class
        y_pred_scores = y_pred[:, int(cls)]
        try:
            auc = roc_auc_score(y_true_binary, y_pred_scores)
            aucs[cls] = auc
        except ValueError:
            pass
    macro_avg = float(np.mean(list(aucs.values())))
    return macro_avg