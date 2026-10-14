import numpy as np
from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import SEED


class TabDPT(TabularModel):

    MODEL_NAME = "TabDPT"
    SHORT_NAME = "dpt"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True

    def initialize_model(self):
        from tabdpt import TabDPTRegressor, TabDPTClassifier
        model_cls = TabDPTClassifier if self.is_cls else TabDPTRegressor
        model = model_cls(device=str(self.device), use_flash=False)
        return model

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        self.model_.fit(x_train, y_train)

    def predict_from_processed(self, x: DataFrame) -> np.ndarray:
        """As the base class, with SEED: tabdpt draws its ensemble, permutations and kNN unseeded otherwise."""
        if not self.is_cls:
            return self.model_.predict(x, seed=SEED)
        probs = self.model_.predict_proba(x, seed=SEED)
        return probs[:, 1] if self.d_output == 2 else probs
