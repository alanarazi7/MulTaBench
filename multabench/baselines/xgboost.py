from dataclasses import dataclass

from xgboost import XGBRegressor, XGBClassifier
from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar.training.devices import CPU_CORES
from multabench.baselines.abstract_model import TabularModel


@dataclass
class XGBoostDefaultHyperparams:
    # Using the "default" hyperparameters of the FT-Transformer paper: https://arxiv.org/pdf/2106.11959
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    booster: str = "gbtree"
    random_state: int = SEED
    nthread: int = CPU_CORES


class XGBoost(TabularModel):

    MODEL_NAME = "XGBoost 🌲"
    SHORT_NAME = "xgb"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = True
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True

    def initialize_model(self) -> XGBRegressor | XGBClassifier:
        model_cls = XGBClassifier if self.is_cls else XGBRegressor
        params = vars(XGBoostDefaultHyperparams())
        return model_cls(**params)

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=self.verbose)
