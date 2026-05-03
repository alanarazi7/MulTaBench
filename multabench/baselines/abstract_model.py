from typing import Tuple, Dict, Optional, Set, List, Any

import numpy as np
import torch
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
from skrub import DatetimeEncoder
from tabstar.preprocessing.dates import fit_date_encoders, transform_date_features
from tabstar.preprocessing.feat_types import detect_numerical_features
from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.sparse import densify_objects
from tabstar.preprocessing.splits import split_to_val

from multabench.baselines.preprocessing.feature_types import transform_feature_types
from multabench.baselines.preprocessing.image_embeddings import (
    transform_image_features,
    fit_image_encoders,
)
from multabench.baselines.training.metrics import calculate_metric, Metrics
from multabench.datasets.all_datasets import MultimodalDatasetID
from multabench.baselines.preprocessing.categorical import fit_categorical_encoders, transform_categorical_features
from multabench.preprocessing.feat_types import classify_semantic_features
from multabench.baselines.preprocessing.numerical import fit_numerical_median, transform_numerical_features
from multabench.baselines.preprocessing.target import transform_preprocess_y, fit_preprocess_y
from multabench.datasets.multimodal import MultimodalError
from multabench.dino.constants import DINOV3_SMALL
from multabench.e5.constants import E5_SMALL_V2
from multabench.baselines.preprocessing.text_embeddings import E5ColumnEncoder, fit_text_encoders, transform_text_features
from multabench.datasets.objects import SupervisedTask
from multabench.baselines.preprocessing.feature_types import detect_image_features
from multabench.utils.warnings import silence_baselines_prints


class TabularModel:

    MODEL_NAME: str
    SHORT_NAME: str
    USE_VAL_SPLIT: bool
    USE_MEDIAN_FILLING: bool
    USE_CATEGORICAL_ENCODING: bool
    USE_TEXT_EMBEDDINGS: bool
    USE_TARGET_ENCODER: bool

    def __init__(self, problem_type: SupervisedTask, device: torch.device,
                 dataset: MultimodalDatasetID | None = None, verbose: bool = False, image_folder: str | None = None,
                 tune_dino: bool = False,
                 dino_train_kwargs: Optional[Dict[str, Any]] = None,
                 dino_model_name: str = DINOV3_SMALL,
                 tune_e5: bool = False,
                 e5_train_kwargs: Optional[Dict[str, Any]] = None,
                 e5_model_name: str = E5_SMALL_V2,
                 pca_components: int = 30,
                 no_pca: bool = False,
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
        self.tune_dino = tune_dino
        self.dino_train_kwargs = dino_train_kwargs or {}
        self.dino_model_name = dino_model_name
        self._tuned_dino_model = None
        self._tuned_dino_processor = None
        self.tune_e5 = tune_e5
        self.e5_train_kwargs = e5_train_kwargs or {}
        self.e5_model_name = e5_model_name
        self.pca_components = pca_components
        self.no_pca = no_pca
        self.model_ = self.initialize_model()
        self.target_transformer: Optional[LabelEncoder] = None
        self.date_transformers: Dict[str, DatetimeEncoder] = {}
        self.image_transformers: Dict[str, Any] = {}  # PCA per image column
        self.numerical_features: Set[str] = set()
        self.numerical_medians: Dict[str, float] = {}
        self.categorical_features: Set[str] = set()
        self.categorical_indices: List[int] = []
        self.categorical_encoders: Dict[str, LabelEncoder] = {}
        self.text_features: Set[str] = set()
        self.text_transformers: Dict[str, E5ColumnEncoder] = {}

    def initialize_model(self):
        raise NotImplementedError("Initialize model method not implemented yet")

    def fit(self, x: DataFrame, y: Series):
        x_train, y_train = x.copy(), y.copy()
        if self.USE_VAL_SPLIT:
            x_train, x_val, y_train, y_val = split_to_val(x=x, y=y, is_cls=self.is_cls)
        else:
            x_val, y_val = None, None
        self.fit_preprocessor(x_train=x_train, y_train=y_train)
        x_train, y_train = self.transform_preprocessor(x=x_train, y=y_train)
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
            self.text_transformers = fit_text_encoders(
                x=x_train,
                text_features=self.text_features,
                device=self.device,
                y=y_train if self.tune_e5 else None,
                tune_e5=self.tune_e5,
                e5_train_kwargs=self.e5_train_kwargs if self.tune_e5 else None,
                is_cls=self.is_cls,
                d_output=self.d_output,
                e5_model_name=self.e5_model_name,
                pca_components=self.pca_components,
                no_pca=self.no_pca,
            )
            self.vprint(f"📝 Detected {len(self.text_transformers)} text features: {sorted(self.text_transformers)}")
        self.fit_internal_preprocessor(x=x_train, y=y_train)

    def transform_preprocessor(self, x: DataFrame, y: Optional[Series]) -> Tuple[DataFrame, Optional[Series]]:
        x, y = densify_objects(x=x, y=y)
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        x = transform_image_features(
            x=x,
            image_transformers=self.image_transformers,
            device=self.device,
            image_folder=self.image_folder,
            dino_model=getattr(self, "_tuned_dino_model", None),
            dino_processor=getattr(self, "_tuned_dino_processor", None),
        )
        image_features = [f"{col}_img_pca_{i}" for col in self.image_transformers.keys() for i in range(self.image_transformers[col].n_components)]
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
            x = transform_text_features(
                x=x,
                text_encoders=self.text_transformers,
                device=self.device,
            )
        return self.transform_internal_preprocessor(x=x, y=y)

    def fit_internal_preprocessor(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        pass

    def transform_internal_preprocessor(self, x: DataFrame, y: Optional[Series]) -> Tuple[DataFrame, Optional[Series]]:
        return x, y

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: Optional[DataFrame], y_val: Optional[Series]):
        raise NotImplementedError("Fit model method not implemented yet")

    def do_model_agnostic_preprocessing(self, x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        raise_if_null_target(y)
        x, y = densify_objects(x=x, y=y)
        self.date_transformers = fit_date_encoders(x=x)
        self.vprint(f"📅 Detected {len(self.date_transformers)} date features: {sorted(self.date_transformers)}")
        # TODO: ConTextTab supports dates natively, perhaps we need to change this to "USE_DATE_TRANSFORMATION"
        x = transform_date_features(x=x, date_transformers=self.date_transformers)
        self.raise_if_no_pca_and_too_multimodal(x=x)
        if self.is_cls:
            self.d_output = len(set(y))
        else:
            self.d_output = 1
        self.image_transformers, self._tuned_dino_model, self._tuned_dino_processor = fit_image_encoders(
            x=x,
            device=self.device,
            image_folder=self.image_folder,
            y=y if self.tune_dino else None,
            is_cls=self.is_cls,
            d_output=self.d_output,
            tune_dino=self.tune_dino,
            dino_train_kwargs=self.dino_train_kwargs if self.tune_dino else None,
            dino_model_name=self.dino_model_name,
            pca_components=self.pca_components,
            no_pca=self.no_pca,
        )
        self.vprint(f"📷 Detected {len(self.image_transformers)} image features: {sorted(self.image_transformers)}")
        non_image_columns = [col for col in x.columns if col not in self.image_transformers]
        x = x[non_image_columns]
        self.numerical_features = detect_numerical_features(x)
        self.vprint(f"🔢 Detected {len(self.numerical_features)} numerical features: {sorted(self.numerical_features)}")
        image_features = [f"{col}_img_pca_{i}" for col in self.image_transformers.keys() for i in range(self.image_transformers[col].n_components)]
        x = transform_feature_types(x=x, numerical_features=self.numerical_features, image_features=image_features)
        semantic_types = classify_semantic_features(x=x, numerical_features=self.numerical_features)
        self.text_features = semantic_types.text_features
        self.categorical_features = semantic_types.categorical_features
        # Indices here are for the current x (no image/text PCA yet). After transform_preprocessor, column
        # order changes (image cols → PCA, text cols → PCA), so indices no longer match. Models that need
        # categorical columns at fit time (e.g. LightGBM) must resolve by name: [c for c in x.columns if c in self.categorical_features].
        self.categorical_indices = [i for i, c in enumerate(x.columns) if c in self.categorical_features]
        if self.USE_TARGET_ENCODER:
            self.target_transformer = fit_preprocess_y(y=y, is_cls=self.is_cls)
        # d_output already set above for fit_image_encoders
        # TODO: drop constant columns, where constants means all values are the same (and no nulls)
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

    def score(self, X, y) -> float:
        metrics = self.score_all_metrics(X=X, y=y)
        return metrics.score

    def score_all_metrics(self, X, y) -> Metrics:
        x = X.copy()
        y = y.copy()
        if self.USE_TARGET_ENCODER:
            y_true = transform_preprocess_y(y=y, scaler=self.target_transformer)
        else:
            y_true = y
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

    def raise_if_no_pca_and_too_multimodal(self, x):
        if self.no_pca:
            _image_cols = set(detect_image_features(x=x))
            _x_no_img = x[[c for c in x.columns if c not in _image_cols]]
            _num_feats = detect_numerical_features(_x_no_img)
            _semantic = classify_semantic_features(x=_x_no_img, numerical_features=_num_feats)
            _n_multimodal = len(_image_cols) + len(_semantic.text_features)
            if _n_multimodal > 5:
                raise MultimodalError(f"Dataset has {_n_multimodal} multimodal features (images + text); skipping for no_pca mode (threshold: 5).")
