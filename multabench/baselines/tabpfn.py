import os
from typing import List

from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import HF_TOKEN, SEED


class TabPFN(TabularModel):
    """The TabPFN versions differ only in VERSION, the name tabpfn's create_default_for_version takes."""

    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True

    VERSION: str

    def initialize_model(self) -> None:
        os.environ["HF_TOKEN"] = HF_TOKEN or ""

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_feature = [i for i, c in enumerate(x_train.columns) if c in self.categorical_features]
        self.model_ = self.get_model_instance(cat_feature=cat_feature)
        self.model_.fit(x_train, y_train)

    def get_model_instance(self, cat_feature: List[int]):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        model_cls = TabPFNRegressor if not self.is_cls else TabPFNClassifier
        return model_cls.create_default_for_version(version=ModelVersion(self.VERSION), random_state=SEED,
                                                    categorical_features_indices=cat_feature,
                                                    ignore_pretraining_limits=True)


class TabPFNv2(TabPFN):

    MODEL_NAME = "TabPFN-v2"
    SHORT_NAME = "tabpfnv2"
    VERSION = "v2"


class TabPFNv2p5(TabPFN):

    MODEL_NAME = "TabPFN-v2p5"
    SHORT_NAME = "tabpfnv2p5"
    VERSION = "v2.5"


class TabPFN3(TabPFN):

    MODEL_NAME = "TabPFN-3"
    SHORT_NAME = "tabpfn3"
    VERSION = "v3"


class TabPFN3p5(TabPFN):

    MODEL_NAME = "TabPFN-3.5"
    SHORT_NAME = "tabpfn3p5"
    VERSION = "v3.5"


class TabPFN3p5Fast(TabPFN):

    MODEL_NAME = "TabPFN-3.5-Fast"
    SHORT_NAME = "tabpfn3p5fast"
    VERSION = "v3.5-fast"
