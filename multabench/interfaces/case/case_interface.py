"""CASE as an interface (Schindler et al., 2026): a tabular model fitted on its usual input plus CASE's context-aware
row embeddings, reduced by PCA, as CaseTransformer's defaults append them to the original columns."""
from typing import Dict, Type

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from sklearn.decomposition import PCA

from multabench.baselines.abstract_model import TabularModel
from multabench.datasets.objects import SupervisedTask
from multabench.embeddings.hub import cached_embeddings
from multabench.interfaces.case.constants import CASE_PCA_COMPONENTS, CASE_SEED
from multabench.interfaces.case.rows import case_rows
from multabench.interfaces.constants import Interface


class CaseInterface(TabularModel):
    """CASE around TABULAR_MODEL, with CaseTransformer's defaults for every model: the model's own input, text embedded
    by the run's text encoder, then 32 PCA components of the row embeddings, fitted on the training rows."""

    TABULAR_MODEL: Type[TabularModel]
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TARGET_ENCODER = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        model = cls.TABULAR_MODEL
        cls.MODEL_NAME, cls.SHORT_NAME = f"CASE-{model.MODEL_NAME}", f"case{model.SHORT_NAME}"
        cls.NEEDS_GPU, cls.ENV, cls.REQUIRED_ENV = model.NEEDS_GPU, model.ENV, model.REQUIRED_ENV
        # The wrapped model embeds the text columns, so a run takes a text encoder as the model's own runs do.
        cls.USE_TEXT_EMBEDDINGS = model.USE_TEXT_EMBEDDINGS

    def __init__(self, problem_type: SupervisedTask, device: torch.device, **kwargs):
        super().__init__(problem_type=problem_type, device=device, **kwargs)
        self.tabular_model = self.TABULAR_MODEL(problem_type=problem_type, device=device, **kwargs)
        self.size, self.fold = kwargs.get("size"), kwargs.get("fold")

    def initialize_model(self) -> None:
        return None

    def fit(self, x: DataFrame, y: Series):
        if self.image_column:
            raise ValueError("CASE runs on text datasets only")
        if self.dataset is None or self.fold is None:
            raise ValueError("CASE reads its embeddings from the fold's cache; give the run's dataset, size and fold")
        self.cache = cached_embeddings(Interface.CASE, self.dataset, size=self.size, fold=self.fold)
        print(f"Reading the CASE embeddings of fold {self.fold} from the cache")
        embeddings = case_rows(self.cache, x)
        # As CaseTransformer.fit: fewer components if there are fewer rows.
        self.pca = PCA(n_components=min(CASE_PCA_COMPONENTS, *embeddings.shape), random_state=CASE_SEED)
        self.tabular_model.fit(self._append(x, self.pca.fit_transform(embeddings)), y)

    def predict(self, x: DataFrame) -> np.ndarray:
        return self.tabular_model.predict(self._transform(x))

    def score_all_metrics(self, X: DataFrame, y: Series) -> Dict:
        return self.tabular_model.score_all_metrics(self._transform(X), y)

    def _transform(self, x: DataFrame) -> DataFrame:
        return self._append(x, self.pca.transform(case_rows(self.cache, x)))

    @staticmethod
    def _append(x: DataFrame, components: np.ndarray) -> DataFrame:
        """The original columns, then CASE's, named as CaseTransformer names them."""
        names = [f"lm_pca_{i}" for i in range(components.shape[1])]
        assert not set(names) & set(x.columns), f"{x.columns} already has CASE's column names"
        return pd.concat([x, DataFrame(components, index=x.index, columns=names)], axis=1)
