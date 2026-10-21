from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import SEED


class TabSTARv1p1(TabularModel):
    """TabSTAR v1.1 at its package defaults; its checkpoints were decontaminated against MulTaBench."""

    MODEL_NAME = "TabSTAR-v1.1"
    SHORT_NAME = "tabstar11"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = False
    USE_TARGET_ENCODER = False

    def initialize_model(self):
        from tabstar.tabstarv1p1_model import TabSTARv1p1Classifier, TabSTARv1p1Regressor
        tabstar_cls = TabSTARv1p1Classifier if self.is_cls else TabSTARv1p1Regressor
        return tabstar_cls(device=self.device, verbose=True, random_state=SEED)

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train)
