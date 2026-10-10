from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import DataFrame, Series

from multabench.constants import SEED
from multabench.baselines.abstract_model import TabularModel
from multabench.utils.devices import CPU_CORES


class LightGBM(TabularModel):

    MODEL_NAME = "LightGBM"
    SHORT_NAME = "light"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True
    NEEDS_GPU = False

    def initialize_model(self) -> LGBMRegressor | LGBMClassifier:
        model_cls = LGBMClassifier if self.is_cls else LGBMRegressor
        params = {"verbose": -1, "random_state": SEED, "n_jobs": CPU_CORES}
        return model_cls(**params)

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        # Use column names so we only mark truly categorical columns; indices can point to PCA cols after transforms.
        cat_cols = [c for c in x_train.columns if c in self.categorical_features]
        self.model_.fit(x_train, y_train, eval_set=[(x_val, y_val)], categorical_feature=cat_cols)
