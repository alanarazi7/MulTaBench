import os
from typing import Tuple, Dict, Optional, Set, List

import numpy as np
import torch
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
from skrub import DatetimeEncoder

from multabench.baselines.preprocessing.dates import fit_date_encoders, transform_date_features
from multabench.baselines.preprocessing.feature_types import detect_feature_types, transform_feature_types
from multabench.baselines.preprocessing.nulls import raise_if_null_target
from multabench.baselines.preprocessing.validation import split_to_val
from multabench.baselines.training.metrics import calculate_metric
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.baselines.preprocessing.categorical import (fit_categorical_encoders, fit_category_dtypes,
                                                            transform_categorical_features)
from multabench.baselines.preprocessing.numerical import fit_numerical_median, transform_numerical_features
from multabench.baselines.preprocessing.target import transform_preprocess_y, fit_preprocess_y
from multabench.benchmark.splits import Size
from multabench.encoders.column_encoders import ColumnEncoder, transform_columns
from multabench.encoders.constants import DEFAULT_IMAGE_ENCODER, DEFAULT_TEXT_ENCODER, ImageEncoder, TextEncoder
from multabench.encoders.registry import make_encoder
from multabench.datasets.objects import SupervisedTask
from multabench.utils.warnings import silence_baselines_prints


class TabularModel:

    MODEL_NAME: str
    SHORT_NAME: str
    USE_VAL_SPLIT: bool
    USE_MEDIAN_FILLING: bool
    USE_CATEGORICAL_ENCODING: bool
    USE_TEXT_EMBEDDINGS: bool
    USE_TARGET_ENCODER: bool
    # Categorical columns as pandas' category dtype, with the training values as categories, for models that read it.
    USE_CATEGORY_DTYPE = False
    NEEDS_GPU = True
    # A model whose requirements conflict with the shared .venv runs in its own, .venv-<ENV>, from requirements-<ENV>.txt.
    ENV: Optional[str] = None
    # Variables a run can't do without, e.g. a licence token; benchmark.py stops before running if one is unset.
    REQUIRED_ENV: Tuple[str, ...] = ()

    @classmethod
    def env_error(cls) -> Optional[str]:
        """Why the model can't run here, e.g. an unset REQUIRED_ENV; benchmark.py stops before running if any."""
        if missing := [v for v in cls.REQUIRED_ENV if not os.getenv(v)]:
            return f"{cls.MODEL_NAME} needs {', '.join(missing)}: set it in .env or pass it to the job"
        return None

    def __init__(self, problem_type: SupervisedTask, device: torch.device,
                 dataset: MulTaBenchDatasetID | None = None, verbose: bool = False, image_folder: str | None = None,
                 image_column: str | None = None,
                 text_encoder: TextEncoder | None = None,
                 image_encoder: ImageEncoder | None = None,
                 fold: int | None = None,
                 size: Size | None = None,
                 **kwargs):
        assert problem_type in {SupervisedTask.REGRESSION, SupervisedTask.BINARY, SupervisedTask.MULTICLASS}
        self.problem_type = problem_type
        self.dataset = dataset
        self.is_cls = bool(problem_type in {SupervisedTask.BINARY, SupervisedTask.MULTICLASS})
        self.device = device
        self.best_val_loss: Optional[None] = None
        self.verbose = verbose
        self.d_output: int = 0
        self.image_folder = image_folder
        self.image_column = image_column
        self.text_encoder = text_encoder or DEFAULT_TEXT_ENCODER
        self.image_encoder = image_encoder or DEFAULT_IMAGE_ENCODER
        context = dict(device=device, dataset=dataset, size=size, fold=fold)
        self.text_embedder = make_encoder(self.text_encoder, **context)
        self.image_embedder = make_encoder(self.image_encoder, image_folder=image_folder, **context)
        self.model_ = self.initialize_model()
        self.target_transformer: Optional[LabelEncoder] = None
        self.date_transformers: Dict[str, DatetimeEncoder] = {}
        self.image_transformers: Dict[str, ColumnEncoder] = {}
        self.numerical_features: Set[str] = set()
        self.numerical_medians: Dict[str, float] = {}
        self.categorical_features: Set[str] = set()
        self.categorical_indices: List[int] = []
        self.categorical_encoders: Dict[str, LabelEncoder] = {}
        self.text_features: Set[str] = set()
        self.text_transformers: Dict[str, ColumnEncoder] = {}
        self._train_embeddings: Dict[str, np.ndarray] = {}

    def initialize_model(self):
        raise NotImplementedError("Initialize model method not implemented yet")

    def fit(self, x: DataFrame, y: Series):
        x_train, y_train = x.copy(), y.copy()
        if self.USE_VAL_SPLIT:
            x_train, x_val, y_train, y_val = split_to_val(x=x, y=y, is_cls=self.is_cls)
        else:
            x_val, y_val = None, None
        self.fit_preprocessor(x_train=x_train, y_train=y_train)
        x_train, y_train = self.transform_preprocessor(x=x_train, y=y_train, train_embeddings=self._train_embeddings)
        self._train_embeddings = {}
        if x_val is not None and y_val is not None:
            x_val, y_val = self.transform_preprocessor(x=x_val, y=y_val)
        self._print_feature_summary(x_train)
        with silence_baselines_prints():
            self.fit_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    def fit_preprocessor(self, x_train: DataFrame, y_train: Series):
        x_train, y_train = self.do_model_agnostic_preprocessing(x=x_train, y=y_train)
        if self.USE_MEDIAN_FILLING:
            self.numerical_medians = fit_numerical_median(x=x_train, numerical_features=self.numerical_features)
        if self.USE_CATEGORICAL_ENCODING:
            self.categorical_encoders = fit_categorical_encoders(x=x_train, categorical_features=self.categorical_features)
        if self.USE_TEXT_EMBEDDINGS:
            self.text_transformers, text_embeddings = self.text_embedder.fit(
                x=x_train, columns=sorted(self.text_features), y=y_train, is_cls=self.is_cls)
            self._train_embeddings.update(text_embeddings)
            self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")
        if self.USE_CATEGORY_DTYPE:
            self.category_dtypes = fit_category_dtypes(x=x_train, categorical_features=self.categorical_features)
        self.fit_internal_preprocessor(x=x_train, y=y_train)

    def transform_preprocessor(self, x: DataFrame, y: Optional[Series],
                               train_embeddings: Optional[Dict[str, np.ndarray]] = None) -> Tuple[DataFrame, Optional[Series]]:
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        x = transform_columns(x=x, encoders=self.image_transformers, train_embeddings=train_embeddings)
        image_features = [f for encoder in self.image_transformers.values() for f in encoder.features]
        x = transform_feature_types(x=x, numerical_features=self.numerical_features, image_features=image_features)
        if y is not None:
            raise_if_null_target(y)
            if self.USE_TARGET_ENCODER:
                y = transform_preprocess_y(y=y, scaler=self.target_transformer)
        if self.USE_MEDIAN_FILLING:
            x = transform_numerical_features(x=x, numerical_medians=self.numerical_medians)
        if self.USE_CATEGORICAL_ENCODING:
            x = transform_categorical_features(x=x, categorical_encoders=self.categorical_encoders)
        if self.USE_TEXT_EMBEDDINGS:
            x = transform_columns(x=x, encoders=self.text_transformers, train_embeddings=train_embeddings)
        if self.USE_CATEGORY_DTYPE:
            x = x.astype(self.category_dtypes)
        return self.transform_internal_preprocessor(x=x, y=y)

    def fit_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        pass

    def transform_internal_preprocessor(self, x: DataFrame, y: Optional[Series]) -> Tuple[DataFrame, Optional[Series]]:
        return x, y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: Optional[DataFrame], y_val: Optional[Series]):
        raise NotImplementedError("Fit model method not implemented yet")

    def do_model_agnostic_preprocessing(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        raise_if_null_target(y)
        feature_types = detect_feature_types(x, image_column=self.image_column)
        self.date_transformers = fit_date_encoders(x=x, date_features=feature_types.date_features)
        self.vprint(f"📅 Detected {len(self.date_transformers)} date features: {sorted(self.date_transformers)}")
        # TODO: a bug for ConTextTab, which reads dates natively: add a USE_DATE_TRANSFORMATION flag so it skips this.
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        if self.is_cls:
            self.d_output = len(set(y))
        else:
            self.d_output = 1
        image_columns = [c for c in x.columns if c in feature_types.image_features]
        self.image_transformers, self._train_embeddings = self.image_embedder.fit(
            x=x, columns=image_columns, y=y, is_cls=self.is_cls)
        self.vprint(f"📷 Detected {len(self.image_transformers)} image features: {sorted(self.image_transformers)}")
        non_image_columns = [col for col in x.columns if col not in self.image_transformers]
        x = x[non_image_columns]
        date_parts = {part for encoder in self.date_transformers.values() for part in encoder.get_feature_names_out()}
        self.numerical_features = feature_types.numerical_features | date_parts
        self.text_features = feature_types.text_features
        self.categorical_features = feature_types.categorical_features
        self.vprint(f"🔢 Detected {len(self.numerical_features)} numerical features: {sorted(self.numerical_features)}")
        image_features = [f for encoder in self.image_transformers.values() for f in encoder.features]
        x = transform_feature_types(x=x, numerical_features=self.numerical_features, image_features=image_features)
        # Indices here are for the current x (no image/text PCA yet). After transform_preprocessor, column
        # order changes (image cols → PCA, text cols → PCA), so indices no longer match. Models that need
        # categorical columns at fit time (e.g. LightGBM) must resolve by name: [c for c in x.columns if c in self.categorical_features].
        self.categorical_indices = [i for i, c in enumerate(x.columns) if c in self.categorical_features]
        self.target_transformer = fit_preprocess_y(y=y, is_cls=self.is_cls)
        # d_output already set above for the image encoder
        return x, y

    def predict(self, x: DataFrame) -> np.ndarray:
        x, _ = self.transform_preprocessor(x=x, y=None)
        return self.predict_from_processed(x=x)

    def predict_from_processed(self, x: DataFrame) -> np.ndarray:
        if not self.is_cls:
            return self.model_.predict(x)
        probs = self.model_.predict_proba(x)
        if self.d_output == 2:
            probs = probs[:, 1]
        return probs

    def score_all_metrics(self, X, y) -> Dict:
        x = X.copy()
        y = y.copy()
        y_true = transform_preprocess_y(y=y, scaler=self.target_transformer)
        y_pred = self.predict(x)
        metrics = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=self.d_output)
        return metrics

    def _print_feature_summary(self, x: DataFrame) -> None:
        cols = list(x.columns)
        n_img = sum(1 for c in cols if "_img_pca_" in c)
        n_txt = sum(1 for c in cols if "_txt_pca_" in c)
        n_tab = len(cols) - n_img - n_txt
        parts = [f"tabular: {n_tab}"]
        if n_img:
            parts.append(f"img_emb: {n_img}")
        if n_txt:
            parts.append(f"txt_emb: {n_txt}")
        print(f"🔭 {len(cols)} features → {self.SHORT_NAME}  [{' | '.join(parts)}]")

    def vprint(self, s: str):
        if self.verbose:
            print(s)
