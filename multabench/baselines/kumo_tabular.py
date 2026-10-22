from unittest.mock import patch

import numpy as np
import torch
from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import SEED

TARGET = "__target__"


class KumoEstimator:
    """Kumo Tabular on pandas, as TabArena and NVIDIA's adapter run it: fit caches the context, predict reads each
    query row from it under float16 autocast."""

    def __init__(self, task: str, size: str, num_estimators: int, n_classes: int, device: torch.device):
        self.task = task
        self.size = size
        self.num_estimators = num_estimators
        self.n_classes = n_classes
        self.device = device

    def _inference(self):
        return torch.amp.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda")

    def fit(self, x: DataFrame, y: Series):
        import sdm
        from sdm.cache import Cache
        self.model = sdm.models.KumoTabular(task=self.task, size=self.size, device=self.device)
        self.stypes = sdm.infer_stypes(x, _low_cardinality="infer")
        x_context = sdm.TableTensor.from_pandas(df=x, stypes=self.stypes, device=self.device)
        target_stype = "numerical" if self.task == "regression" else "categorical"
        y_context = sdm.TableTensor.from_pandas(df=y.rename(TARGET).to_frame(), stypes={TARGET: target_stype},
                                                device=self.device)
        rows, cols = x_context.shape[-2:]
        # NVIDIA's adapter fits one ensemble member at a time above these sizes, to bound GPU memory.
        batch_size = 1 if rows > 2_000 or rows * cols >= 50_000 else self.num_estimators
        generator = torch.Generator(self.device).manual_seed(SEED)
        # sdm copies each estimator's fitted cache to pinned CPU memory, up to 56 GB on Hearthstone, past the 32 GB of
        # RAM; Cache._apply_tensor does only that copy in fit, so skipping it keeps the caches on the GPU, where predict
        # reads them anyway.
        with torch.inference_mode(), self._inference(), patch.object(Cache, "_apply_tensor", lambda c, fn: c):
            self.model.fit(x=x_context, y=y_context, num_estimators=self.num_estimators,
                           estimator_batch_size=batch_size, generator=generator)
        return self

    def _forward(self, x: DataFrame):
        import sdm
        x_query = sdm.TableTensor.from_pandas(df=x, stypes=self.stypes, device=self.device)
        with torch.inference_mode(), self._inference():
            return self.model.predict(x_query)

    def predict(self, x: DataFrame) -> np.ndarray:
        """The mean of the predicted quantiles."""
        return self._forward(x).numerical.float().mean(dim=-1).cpu().numpy()

    def predict_proba(self, x: DataFrame) -> np.ndarray:
        import sdm
        out = self._forward(x)
        probs = np.zeros((len(x), self.n_classes), dtype=np.float32)
        probs[:, [int(label) for label in out.columns[sdm.Stype.numerical]]] = out.numerical.float().cpu().numpy()
        return probs


class Kumo(TabularModel):
    """NVIDIA's Kumo Tabular; its sizes differ only in the checkpoint (SIZE) and TabArena's NUM_ESTIMATORS."""

    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True

    SIZE: str
    NUM_ESTIMATORS: int

    def initialize_model(self) -> None:
        return None

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        task = "classification" if self.is_cls else "regression"
        self.model_ = KumoEstimator(task=task, size=self.SIZE, num_estimators=self.NUM_ESTIMATORS,
                                    n_classes=self.d_output, device=self.device)
        self.model_.fit(x_train, y_train)


class KumoTabular(Kumo):

    MODEL_NAME = "Kumo-Tabular"
    SHORT_NAME = "kumo"
    SIZE = "large"
    NUM_ESTIMATORS = 16


class KumoTabularMedium(Kumo):

    MODEL_NAME = "Kumo-Tabular-M"
    SHORT_NAME = "kumomedium"
    SIZE = "medium"
    NUM_ESTIMATORS = 8


class KumoTabularSmall(Kumo):

    MODEL_NAME = "Kumo-Tabular-S"
    SHORT_NAME = "kumosmall"
    SIZE = "small"
    NUM_ESTIMATORS = 8
