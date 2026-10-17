import json
from importlib import resources
from pathlib import Path

import numpy as np
from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import SEED
from multabench.utils.hardware import visible_cpus

HF_REPO = "stable-ai/LimiX-2"
WEIGHTS = "LimiX-2.ckpt"
REVISION = "20c07a07801973a0aec57c31062bdb2ea1cda2b2"
CONFIGS = {True: "cls_default_noretrieval_v2.json", False: "reg_default_noretrieval_v2.json"}
HEAD_CLASSES = 10


def _patch_limix():
    """TabArena's two patches: LimiX's cache defaults to a path on the developers' cluster, and ARPACK can fail."""
    from inference.v2_0 import preprocess
    from inference.v2_0.predictor import LimiXPredictor
    from scipy.sparse.linalg import ArpackError
    from sklearn.decomposition import TruncatedSVD

    cache_cls = LimiXPredictor.CacheManager
    if getattr(cache_cls, "_multabench_patched", False):
        return
    cache_init = cache_cls.__init__

    def init_cache(self, cache_dir="/mnt/public/infe_cache", *args, **kwargs):
        try:
            cache_init(self, cache_dir, *args, **kwargs)
        except OSError:
            fallback = Path.home() / ".cache" / "limix" / "infe_cache"
            fallback.mkdir(parents=True, exist_ok=True)
            cache_init(self, str(fallback), *args, **kwargs)

    class SVDWithFallback(TruncatedSVD):
        def fit_transform(self, x, y=None):
            try:
                return super().fit_transform(x, y)
            except ArpackError:
                self.algorithm = "randomized"
                return super().fit_transform(x, y)

    cache_cls.__init__ = init_cache
    cache_cls._multabench_patched = True
    preprocess.TruncatedSVD = SVDWithFallback


def _predictor(device: str, is_cls: bool):
    import torch
    from huggingface_hub import hf_hub_download
    from inference.v2_0.predictor import LimiXPredictor
    _patch_limix()
    config = json.loads(resources.files("config").joinpath(CONFIGS[is_cls]).read_text())
    return LimiXPredictor(device=torch.device(device), model_path=hf_hub_download(HF_REPO, WEIGHTS, revision=REVISION),
                          inference_config=config, preprocess_num_jobs=visible_cpus())


def _numpy(preds) -> np.ndarray:
    import torch
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().float().cpu().numpy()
    return np.asarray(preds, dtype=np.float32)


class LimiXCodeRow:
    """ManyClassClassifier's base estimator, as in TabArena: LimiX on one codebook row, rebuilding the frame."""

    def __init__(self, device: str, dtypes: Series):
        self.device = device
        self.dtypes = dtypes

    def __sklearn_clone__(self):
        return LimiXCodeRow(device=self.device, dtypes=self.dtypes)

    def _frame(self, x) -> DataFrame:
        return DataFrame(x, columns=self.dtypes.index).astype(self.dtypes)

    def fit(self, x, y):
        self.predictor = _predictor(self.device, is_cls=True)
        self.x_train, self.y_train = self._frame(x), np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def predict_proba(self, x) -> np.ndarray:
        return _numpy(self.predictor.predict(self.x_train, self.y_train, self._frame(x), task_type="Classification"))


class LimiX2(TabularModel):
    """Stable AI's LimiX-2 at TabArena's default, with tabpfn-extensions' output coding above 10 classes; it needs
    Python 3.12, so it runs in its own venv (requirements-limix2.in)."""

    MODEL_NAME = "LimiX-2"
    SHORT_NAME = "limix2"
    ENV = "limix2"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True
    USE_CATEGORY_DTYPE = True

    def initialize_model(self) -> None:
        return None

    def transform_internal_preprocessor(self, x: DataFrame, y: Series | None):
        # LimiX encodes categoricals itself, from object columns.
        categories = [c for c in x.columns if x[c].dtype == "category"]
        return x.astype(dict.fromkeys(categories, object)), y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        if self.d_output > HEAD_CLASSES:
            from tabpfn_extensions.many_class import ManyClassClassifier
            self.model_ = ManyClassClassifier(estimator=LimiXCodeRow(device=str(self.device), dtypes=x_train.dtypes),
                                              alphabet_size=HEAD_CLASSES, random_state=SEED)
            self.model_.fit(x_train, y_train)
            return
        self.model_ = _predictor(str(self.device), is_cls=self.is_cls)
        self.x_train, self.y_train = x_train, y_train.to_numpy()

    def predict_from_processed(self, x: DataFrame) -> np.ndarray:
        if self.d_output > HEAD_CLASSES:
            return self.model_.predict_proba(x)
        task_type = "Classification" if self.is_cls else "Regression"
        preds = _numpy(self.model_.predict(self.x_train, self.y_train, x, task_type=task_type))
        return preds[:, 1] if self.d_output == 2 else preds
