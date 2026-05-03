from dataclasses import dataclass

from pandas import DataFrame, Series
from pytabkit import TabM_D_Classifier, TabM_D_Regressor

from tabstar.constants import SEED
from tabstar.training.devices import CPU_CORES
from multabench.baselines.abstract_model import TabularModel


@dataclass
class TabMDefaultHyperparams:
    device: str
    random_state: int = SEED
    n_threads: int = CPU_CORES


class TabM(TabularModel):
    MODEL_NAME = "TabM Ⓜ️"
    SHORT_NAME = "tabm"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = True
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True

    def initialize_model(self) -> TabM_D_Classifier | TabM_D_Regressor:
        params = TabMDefaultHyperparams(device=str(self.device))
        params = vars(params)
        model_cls = TabM_D_Classifier if self.is_cls else TabM_D_Regressor
        return model_cls(**params)

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_col_names = list(self.categorical_features)
        self.model_.fit(x_train, y_train, x_val, y_val, cat_col_names=cat_col_names)
