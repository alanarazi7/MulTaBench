import os

from pandas import DataFrame, Series
from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import HF_TOKEN


class ConTextTab(TabularModel):
    MODEL_NAME = "ConTextTab 🏢"
    SHORT_NAME = "ctx"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    # TODO: Should we use text embeddings for ConTextTab?
    USE_TEXT_EMBEDDINGS = False
    USE_TARGET_ENCODER = False

    def initialize_model(self) -> SAP_RPT_OSS_Classifier | SAP_RPT_OSS_Regressor:
        os.environ["HF_TOKEN"] = HF_TOKEN
        model_cls = SAP_RPT_OSS_Classifier if self.is_cls else SAP_RPT_OSS_Regressor
        # Initialize the regressor, 8k context and 8-fold bagging gives best performance, reduce if running out of memory
        model = model_cls(max_context_size=8192, bagging=8)
        return model

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        # TODO: ConTextTab natively supports dates, perhaps we need to process them differently
        self.model_.fit(x_train, y_train)

