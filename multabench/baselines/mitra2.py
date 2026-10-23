import os
import shutil
import tempfile

import numpy as np
from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.baselines.preprocessing.validation import split_to_val
from multabench.constants import SEED
from multabench.datasets.objects import SupervisedTask
from multabench.utils.hardware import visible_cpus

LABEL = "__target__"
HEAD_CLASSES = 10
BAG_FOLDS = 8
# Out of GPU memory, predict may halve its context to 512 rows as fine-tuning does, not stop at TabArena's 8,192.
PREDICT_SUPPORT_FLOOR = 512
PROBLEM_TYPES = {SupervisedTask.BINARY: "binary", SupervisedTask.MULTICLASS: "multiclass",
                 SupervisedTask.REGRESSION: "regression"}
METRICS = {SupervisedTask.BINARY: "roc_auc", SupervisedTask.MULTICLASS: "log_loss",
           SupervisedTask.REGRESSION: "root_mean_squared_error"}


class MitraV2(TabularModel):
    """Amazon's Mitra-v2 at TabArena's default: TabArena's wrapper, bagged by AutoGluon in 8 folds, each fine-tuned and
    predicting with all training rows as context, in its own venv (requirements-mitra2.in). Above its 10-class head it
    runs the official mitra-finetune's class hierarchy, which fits the bag first at prediction time."""

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
            return self._fit_hierarchy(x_train, y_train)
        from autogluon.tabular import TabularPredictor
        from tabarena.models.mitra_v2.model import MitraV2Model
        self.path_ = tempfile.mkdtemp(prefix="mitra2_")
        self.model_ = TabularPredictor(label=LABEL, problem_type=PROBLEM_TYPES[self.problem_type],
                                       eval_metric=METRICS[self.problem_type], path=self.path_, verbosity=0)
        # As in TabArena, each of the bag's children gets its own seed.
        seeds = {"model_random_seed": SEED, "vary_seed_across_folds": True}
        params = {"ag_args_ensemble": seeds, "predict_support_floor": PREDICT_SUPPORT_FLOOR}
        self.model_.fit(x_train.assign(**{LABEL: y_train.to_numpy()}),
                        hyperparameters={MitraV2Model: params}, num_bag_folds=BAG_FOLDS, num_bag_sets=1,
                        fit_weighted_ensemble=False, calibrate=False, num_cpus=visible_cpus(), num_gpus=1)

    def _fit_hierarchy(self, x_train: DataFrame, y_train: Series):
        """mitra-finetune's balanced class tree of native-head nodes, which validate on a stratified split."""
        from huggingface_hub import snapshot_download
        from mitra_finetune import MitraFinetune
        from tabarena.models.mitra_v2._internal.recipe import (FINE_TUNE_BUDGET_S, HF_CLASSIFIER_REPO,
                                                               HF_CLASSIFIER_REVISION)
        # The fine-tuning budget TabArena's wrapper applies; mitra-finetune leaves it off by default.
        os.environ["MITRA_FT_BUDGET_S"] = str(int(FINE_TUNE_BUDGET_S))
        x_fit, x_val, y_fit, y_val = split_to_val(x=x_train, y=y_train, is_cls=True)
        checkpoint = snapshot_download(HF_CLASSIFIER_REPO, revision=HF_CLASSIFIER_REVISION)
        self.hierarchy_ = MitraFinetune(checkpoint, eval_metric=METRICS[self.problem_type], random_state=SEED,
                                        num_bag_folds=BAG_FOLDS)
        self.hierarchy_.fit(x_fit, y_fit.to_numpy(), X_val=x_val, y_val=y_val.to_numpy())

    def predict_from_processed(self, x: DataFrame) -> np.ndarray:
        if hasattr(self, "hierarchy_"):
            return self.hierarchy_.predict_proba(x)
        if not self.is_cls:
            preds = self.model_.predict(x).to_numpy()
        else:
            probs = self.model_.predict_proba(x).to_numpy()
            preds = probs[:, 1] if self.d_output == 2 else probs
        shutil.rmtree(self.path_, ignore_errors=True)
        return preds
