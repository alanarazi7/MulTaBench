import numpy as np
from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import SEED

# TabArena reads the weights at main; this is main as of TabArena's runs and ours.
REVISION = "093ad1c2613e1e95d26df8610451d2e02ed53cbe"


def _float32(x: DataFrame) -> np.ndarray:
    values = x.to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = np.nan
    return values


class EXAONETabular(TabularModel):
    """LG's EXAONE Tabular at TabArena's default: 8 ensemble members in float16, in its own venv
    (requirements-exaone.in)."""

    MODEL_NAME = "EXAONE-Tabular"
    SHORT_NAME = "exaone"
    ENV = "exaone"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True

    def initialize_model(self) -> None:
        return None

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        from exaonetabular import EXAONETabularClassifier, EXAONETabularRegressor
        model_cls = EXAONETabularClassifier if self.is_cls else EXAONETabularRegressor
        self.model_ = model_cls.from_pretrained(device=str(self.device), revision=REVISION, ensemble_count=8,
                                                compute_dtype="float16", seed=SEED)
        self.model_.fit(_float32(x_train), y_train.to_numpy())

    def predict_from_processed(self, x: DataFrame) -> np.ndarray:
        return super().predict_from_processed(_float32(x))
