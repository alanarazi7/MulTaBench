import numpy as np
import pytest
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from multabench.baselines.autogluon_mm import TASK2METRIC
from multabench.baselines.training.metrics import calculate_metric, get_scorer
from multabench.datasets.objects import SupervisedTask
from multabench.result_keys import METRIC, TEST_ERROR, TEST_SCORE

TASK2D_OUTPUT = {SupervisedTask.REGRESSION: 1, SupervisedTask.BINARY: 2, SupervisedTask.MULTICLASS: 3}


@pytest.mark.parametrize("task", list(TASK2D_OUTPUT))
def test_autogluon_mm_optimizes_the_scored_metric(task):
    assert TASK2METRIC[task] == get_scorer(TASK2D_OUTPUT[task]).name


def test_regression_error_is_rmse():
    y_true, y_pred = np.array([1.0, 2.0, 4.0]), np.array([1.5, 2.0, 3.0])
    m = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=1)
    assert m[METRIC] == "root_mean_squared_error"
    assert m[TEST_ERROR] == pytest.approx(np.sqrt(mean_squared_error(y_true, y_pred)))
    assert m[TEST_SCORE] == pytest.approx(-m[TEST_ERROR])


@pytest.mark.parametrize("two_columns", [False, True])
def test_binary_error_is_one_minus_auc(two_columns):
    y_true, p = np.array([0, 1, 0, 1, 1]), np.array([0.2, 0.7, 0.6, 0.4, 0.9])
    y_pred = np.column_stack([1 - p, p]) if two_columns else p
    m = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=2)
    assert m[METRIC] == "roc_auc"
    assert m[TEST_SCORE] == pytest.approx(roc_auc_score(y_true, p))
    assert m[TEST_ERROR] == pytest.approx(1 - m[TEST_SCORE])


def test_multiclass_error_is_log_loss():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3]])
    m = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=3)
    assert m[METRIC] == "log_loss"
    assert m[TEST_ERROR] == pytest.approx(log_loss(y_true, y_pred))
