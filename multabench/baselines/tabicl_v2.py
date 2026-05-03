from pandas import DataFrame, Series
from tabicl import TabICLClassifier, TabICLRegressor

from multabench.baselines.abstract_model import TabularModel


class TabICLv2(TabularModel):

    MODEL_NAME = "TabICLv2 🗼"
    SHORT_NAME = "iclv2"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True

    def initialize_model(self):
        cls = TabICLClassifier if self.is_cls else TabICLRegressor
        return cls(device=str(self.device))

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train)
