import numpy as np
from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import SEED

HEAD_CLASSES = 10


class TabFMCodeRow:
    """ManyClassClassifier's base estimator, as in TabArena: a TabFM classifier per codebook row, sharing the network
    and rebuilding the frame from the arrays the codebook passes."""

    def __init__(self, network, dtypes: Series):
        self.network = network
        self.dtypes = dtypes

    def __sklearn_clone__(self):
        return TabFMCodeRow(network=self.network, dtypes=self.dtypes)

    def _frame(self, x) -> DataFrame:
        return DataFrame(x, columns=self.dtypes.index).astype(self.dtypes)

    def fit(self, x, y):
        from tabfm import TabFMClassifier
        self.model = TabFMClassifier(model=self.network, random_state=SEED).fit(self._frame(x), np.asarray(y))
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, x) -> np.ndarray:
        return self.model.predict_proba(self._frame(x))


class TabFM(TabularModel):
    """Google's TabFM at TabArena's default, with tabpfn-extensions' output coding above its 10-class head."""

    MODEL_NAME = "TabFM"
    SHORT_NAME = "tabfm"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True
    USE_CATEGORY_DTYPE = True

    def initialize_model(self) -> None:
        return None

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch
        network = tabfm_v1_0_0_pytorch.load(model_type="classification" if self.is_cls else "regression",
                                            device=str(self.device))
        if self.d_output > HEAD_CLASSES:
            from tabpfn_extensions.many_class import ManyClassClassifier
            self.model_ = ManyClassClassifier(estimator=TabFMCodeRow(network=network, dtypes=x_train.dtypes),
                                              alphabet_size=HEAD_CLASSES, random_state=SEED)
        else:
            cls = TabFMClassifier if self.is_cls else TabFMRegressor
            self.model_ = cls(model=network, random_state=SEED)
        self.model_.fit(x_train, y_train)
