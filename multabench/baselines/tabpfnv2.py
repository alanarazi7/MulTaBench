import os
from typing import List

from pandas import DataFrame, Series
from tabpfn.constants import ModelVersion

from tabstar.constants import SEED
from multabench.baselines.abstract_model import TabularModel
from tabpfn import TabPFNClassifier, TabPFNRegressor

from multabench.constants import HF_TOKEN


class TabPFNv2(TabularModel):

    MODEL_NAME = "TabPFN-v2 🤯"
    SHORT_NAME = "tabpfnv2"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True

    MAX_ROWS = 10000
    MAX_COLS = 500

    def initialize_model(self) -> None:
        os.environ["HF_TOKEN"] = HF_TOKEN or ""

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        if len(x_train) > self.MAX_ROWS:
            raise RuntimeError(f"TabPFN is not designed to handle datasets larger than {self.MAX_ROWS} samples. ")
        if len(x_train.columns) > self.MAX_COLS:
            print(f"🚨 Warning: {len(x_train.columns)} features detected, ignoring pretraining limits.")
        cat_feature = [i for i, c in enumerate(x_train.columns) if c in self.categorical_features]
        self.model_ = self.get_model_instance(cat_feature=cat_feature)
        self.model_.fit(x_train, y_train)

    def get_model_instance(self, cat_feature: List[int]) -> TabPFNClassifier | TabPFNRegressor:
        model_cls = TabPFNRegressor if not self.is_cls else TabPFNClassifier
        return model_cls.create_default_for_version(version=ModelVersion.V2, random_state=SEED,
                                                           categorical_features_indices=cat_feature,
                                                           ignore_pretraining_limits=True)


class TabPFNv2p5(TabPFNv2):

    MODEL_NAME = "TabPFN-v2p5 🇩🇪"
    SHORT_NAME = "tabpfnv2p5"

    MAX_ROWS = 50000
    MAX_COLS = 2000

    def get_model_instance(self, cat_feature: List[int]) -> TabPFNClassifier | TabPFNRegressor:
        model_cls = TabPFNRegressor if not self.is_cls else TabPFNClassifier
        return model_cls.create_default_for_version(version=ModelVersion.V2_5, random_state=SEED,
                                                    categorical_features_indices=cat_feature,
                                                    ignore_pretraining_limits=True)

