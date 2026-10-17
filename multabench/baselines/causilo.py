from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import SEED


class Causilo(TabularModel):
    """Nums AI's Causilo at TabArena's default; it needs torch>=2.13, so it runs in its own venv."""

    MODEL_NAME = "Causilo"
    SHORT_NAME = "causilo"
    ENV = "causilo"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True
    USE_CATEGORY_DTYPE = True

    def initialize_model(self):
        from causilo import CausiloClassifier, CausiloRegressor
        cls = CausiloClassifier if self.is_cls else CausiloRegressor
        return cls(n_estimators=8, random_state=SEED, device=str(self.device))

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train)
