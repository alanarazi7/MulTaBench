import shutil
import tempfile

import numpy as np
from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.constants import SEED
from multabench.datasets.objects import SupervisedTask
from multabench.utils.hardware import visible_cpus

LABEL = "__target__"
HEAD_CLASSES = 10
BAG_FOLDS = 8
PROBLEM_TYPES = {SupervisedTask.BINARY: "binary", SupervisedTask.MULTICLASS: "multiclass",
                 SupervisedTask.REGRESSION: "regression"}
METRICS = {SupervisedTask.BINARY: "roc_auc", SupervisedTask.MULTICLASS: "log_loss",
           SupervisedTask.REGRESSION: "root_mean_squared_error"}


class MitraV2(TabularModel):
    """Amazon's Mitra-v2 at TabArena's default: TabArena's wrapper, bagged by AutoGluon in 8 folds, each fine-tuned and
    predicting with all training rows as context, in its own venv (requirements-mitra2.in)."""

    MODEL_NAME = "Mitra-v2"
    SHORT_NAME = "mitra2"
    ENV = "mitra2"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True
    USE_TARGET_ENCODER = True
    USE_CATEGORY_DTYPE = True

    def initialize_model(self) -> None:
        return None

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        if self.d_output > HEAD_CLASSES:
            raise ValueError(f"Mitra-v2 supports at most {HEAD_CLASSES} classes, not {self.d_output}, as in TabArena")
        from autogluon.tabular import TabularPredictor
        from tabarena.models.mitra_v2.model import MitraV2Model
        self.path_ = tempfile.mkdtemp(prefix="mitra2_")
        self.model_ = TabularPredictor(label=LABEL, problem_type=PROBLEM_TYPES[self.problem_type],
                                       eval_metric=METRICS[self.problem_type], path=self.path_, verbosity=0)
        # As in TabArena, each of the bag's children gets its own seed.
        seeds = {"model_random_seed": SEED, "vary_seed_across_folds": True}
        self.model_.fit(x_train.assign(**{LABEL: y_train.to_numpy()}),
                        hyperparameters={MitraV2Model: {"ag_args_ensemble": seeds}}, num_bag_folds=BAG_FOLDS,
                        num_bag_sets=1, fit_weighted_ensemble=False, calibrate=False, num_cpus=visible_cpus(),
                        num_gpus=1)

    def predict_from_processed(self, x: DataFrame) -> np.ndarray:
        if not self.is_cls:
            preds = self.model_.predict(x).to_numpy()
        else:
            probs = self.model_.predict_proba(x).to_numpy()
            preds = probs[:, 1] if self.d_output == 2 else probs
        shutil.rmtree(self.path_, ignore_errors=True)
        return preds
