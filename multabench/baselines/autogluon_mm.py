import hashlib
import os
import shutil
import tempfile
import time

import numpy as np
from pandas import DataFrame, Series, notna

from autogluon.multimodal import MultiModalPredictor

from tabstar.preprocessing.feat_types import detect_numerical_features
from multabench.baselines.preprocessing.target import fit_preprocess_y, transform_preprocess_y
from multabench.baselines.abstract_model import TabularModel
from multabench.baselines.preprocessing.feature_types import detect_image_features
from multabench.datasets.objects import SupervisedTask
from multabench.e5.constants import E5_SMALL_V2
from multabench.preprocessing.feat_types import classify_semantic_features

# DINOv2 (timm format) — DINOv3 is not supported by timm/AutoGluon as of Apr 2026
AGMM_IMAGE_MODEL = "vit_small_patch14_dinov2.lvd142m"
AGMM_LABEL = "__target__"

TASK2AG = {
    SupervisedTask.BINARY: "binary",
    SupervisedTask.MULTICLASS: "multiclass",
    SupervisedTask.REGRESSION: "regression",
}

TASK2METRIC = {
    SupervisedTask.BINARY: "roc_auc",
    SupervisedTask.MULTICLASS: "roc_auc_ovr",
    SupervisedTask.REGRESSION: "r2",
}


def unique_run_dir(dataset_name: str, x: DataFrame) -> str:
    ts = int(time.time())
    data_hash = hashlib.md5(x.head(5).to_csv().encode()).hexdigest()[:8]
    return tempfile.mkdtemp(prefix=f"agmm_{dataset_name}_{ts}_{data_hash}_")


class AutoGluonMM(TabularModel):

    MODEL_NAME = "AutoGluon-MM 🧴"
    SHORT_NAME = "agmm"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = False
    USE_TARGET_ENCODER = True

    def initialize_model(self):
        return None

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        raise NotImplementedError("AutoGluonMM uses a custom fit() — fit_model() is never called.")

    def fit(self, x: DataFrame, y: Series):
        # Fit target encoder for consistent scoring with other baselines
        self.target_transformer = fit_preprocess_y(y=y, is_cls=self.is_cls)
        y_enc = transform_preprocess_y(y=y, scaler=self.target_transformer)
        if self.is_cls:
            self.d_output = len(set(y_enc))
        else:
            self.d_output = 1

        # Detect modalities
        image_cols = detect_image_features(x)
        numerical = detect_numerical_features(x)
        semantic = classify_semantic_features(x=x, numerical_features=numerical)
        text_cols = semantic.text_features

        # Expand image file paths
        x_ag = x.copy()
        if image_cols and self.image_folder:
            for col in image_cols:
                x_ag[col] = x_ag[col].apply(lambda f: os.path.join(self.image_folder, str(f)) if notna(f) else f)

        # Build base model config based on detected modalities
        has_image = bool(image_cols and self.image_folder)
        has_text = bool(text_cols)
        hparams = {}
        if has_image:
            hparams["model.timm_image.checkpoint_name"] = AGMM_IMAGE_MODEL
        if has_text:
            hparams["model.hf_text.checkpoint_name"] = E5_SMALL_V2

        df_train = x_ag.copy()
        df_train[AGMM_LABEL] = y_enc.values

        output_dir = unique_run_dir(dataset_name=getattr(self.dataset, "name", "unknown"), x=x)
        self.predictor_ = MultiModalPredictor(
            label=AGMM_LABEL,
            problem_type=TASK2AG[self.problem_type],
            eval_metric=TASK2METRIC[self.problem_type],
            path=output_dir,
            pretrained=True,
        )
        self.predictor_.fit(df_train, hyperparameters=hparams, clean_ckpts=True)

    def _prepare_x(self, x: DataFrame) -> DataFrame:
        x_ag = x.copy()
        image_cols = detect_image_features(x_ag)
        if image_cols and self.image_folder:
            for col in image_cols:
                x_ag[col] = x_ag[col].apply(lambda f: os.path.join(self.image_folder, str(f)) if notna(f) else f)
        return x_ag

    def predict(self, x: DataFrame) -> np.ndarray:
        x_ag = self._prepare_x(x)
        if not self.is_cls:
            result = self.predictor_.predict(x_ag).to_numpy()
        else:
            proba = self.predictor_.predict_proba(x_ag).to_numpy()
            result = proba[:, 1] if self.d_output == 2 else proba
        shutil.rmtree(self.predictor_.path, ignore_errors=True)
        self.predictor_ = None
        return result
