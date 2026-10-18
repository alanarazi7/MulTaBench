import os
from dataclasses import dataclass

from pandas import DataFrame, Series

from multabench.constants import SEED
from multabench.utils.devices import CPU_CORES
from multabench.baselines.abstract_model import TabularModel
from multabench.datasets.objects import SupervisedTask


@dataclass
class RealMlpDefaultHyperparams:
    device: str
    val_metric_name: str
    random_state: int = SEED
    n_threads: int = CPU_CORES
    use_ls: bool = False

class RealMLP(TabularModel):
    MODEL_NAME = "RealMLP"
    SHORT_NAME = "real"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = True
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True

    def initialize_model(self):
        import torch
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor
        # pytabkit's categorical layers use CUDA scatter kernels, which made reruns differ by up to 3.6%.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        # The epoch is picked on the leaderboard's metric (ROC AUC, log loss, RMSE), as TabArena's RealMLP does.
        task2metric = {SupervisedTask.BINARY: '1-auc_ovr_alt',
                       SupervisedTask.MULTICLASS: 'cross_entropy',
                       SupervisedTask.REGRESSION: 'rmse'}
        val_metric = task2metric[self.problem_type]
        params = RealMlpDefaultHyperparams(device=str(self.device), val_metric_name=val_metric)
        params = vars(params)
        model_cls = RealMLP_TD_Classifier if self.is_cls else RealMLP_TD_Regressor
        return model_cls(**params)

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_col_names = list(self.categorical_features)
        self.model_.fit(x_train, y_train, x_val, y_val, cat_col_names=cat_col_names)
