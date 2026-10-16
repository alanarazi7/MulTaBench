from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import SEED
from multabench.utils.devices import CPU_CORES

# The Hub revision of TabArena's results; the regression checkpoint was replaced after it.
HF_REPO = "occams/Xiaomi-TabLDM"
REVISION = "0e10427273bc862fb60fe7df84c89e5298e16cc7"
# The classifier defaults of the commit TabArena's results ran, which upstream changed since.
CLASSIFIER_DEFAULTS = {"n_estimators": 8, "feat_shuffle_method": "latin", "cat_random_encode": False, "use_amp": "auto"}


class XiaomiTabLDM(TabularModel):
    """Xiaomi's TabLDM at TabArena's default: its plain ensemble, without the candidate enhancement, whose rank
    normalizer fails when a predict batch has more rows than the training set."""

    MODEL_NAME = "Xiaomi-TabLDM"
    SHORT_NAME = "tabldm"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True
    USE_CATEGORY_DTYPE = True

    def initialize_model(self) -> None:
        return None

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        from huggingface_hub import hf_hub_download
        from tabldm import TabLDMClassifier, TabLDMRegressor
        checkpoint = "checkpoints/clf_default.ckpt" if self.is_cls else "checkpoints/reg_default.ckpt"
        model_cls, defaults = (TabLDMClassifier, CLASSIFIER_DEFAULTS) if self.is_cls else (TabLDMRegressor, {})
        self.model_ = model_cls(device=str(self.device), n_jobs=CPU_CORES, random_state=SEED, enhance_candidates=False,
                                model_path=hf_hub_download(HF_REPO, checkpoint, revision=REVISION), **defaults)
        self.model_.fit(x_train, y_train)
