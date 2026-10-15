import numpy as np
import torch
from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import SEED

# The weights TabArena benchmarked; tabdpt 1.3 can't load the older TabDPT's, nor 1.1 these.
HF_REPO = "Layer6/TabDPT"
WEIGHTS = "tabdpt1_3.safetensors"
REVISION = "a5ca6e01c0fa09ec68c73e958e5199d1932abb3a"


class TabDPT13(TabularModel):
    """TabDPT 1.3 at TabArena's default; tabdpt 1.3 conflicts with the 1.1 TabDPT needs, so it has its own venv."""

    MODEL_NAME = "TabDPT-1.3"
    SHORT_NAME = "dpt13"
    ENV = "tabdpt13"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True

    def initialize_model(self) -> None:
        return None

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        from huggingface_hub import hf_hub_download
        from tabdpt import TabDPTClassifier, TabDPTRegressor
        model_cls = TabDPTClassifier if self.is_cls else TabDPTRegressor
        self.model_ = model_cls(device=str(self.device), use_flash=True, compile=False, verbose=False,
                                model_weight_path=hf_hub_download(HF_REPO, WEIGHTS, revision=REVISION),
                                normalizer="standard", missing_indicators=False, clip_sigma=8,
                                feature_reduction="pca", context_reduction="subsample", faiss_metric="l2")
        # tabdpt reduces wide tables with torch.pca_lowrank, which draws from torch's global generator.
        torch.manual_seed(SEED)
        self.model_.fit(x_train.to_numpy(), y_train.to_numpy())

    def predict_from_processed(self, x: DataFrame) -> np.ndarray:
        if not self.is_cls:
            return self.model_.predict(x.to_numpy(), seed=SEED)
        probs = self.model_.ensemble_predict_proba(x.to_numpy(), seed=SEED)
        return probs[:, 1] if self.d_output == 2 else probs
