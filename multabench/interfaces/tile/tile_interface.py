"""TILE as an interface (Lefebvre et al., 2026): a tabular model fitted on the input TILE's own TILEFeatures builds,
every string column embedded in the context of its row and reduced into the model's feature window, next to the
features that bypass the LLM."""
import importlib.util
from typing import Dict, Optional, Type

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series

from multabench.baselines.abstract_model import TabularModel
from multabench.datasets.objects import SupervisedTask
from multabench.embeddings.hub import cached_embeddings
from multabench.interfaces.constants import Interface
from multabench.interfaces.tile.rows import TileRows, tile_string_columns, tile_table


class TileInterface(TabularModel):
    """TILE around TABULAR_MODEL, which gets TILEFeatures' input: each string column's PCA, the numbers and dates, the
    target encoding of the string columns whose values repeat, out of fold on the training rows, and, if its learner
    takes them, the low-cardinality ones as categories. TILE_LEARNER is its learner in tile.learners.LEARNERS, whose
    feature window it takes; without one, TILE's default, 30 components per string column, meant for non-PFN models."""

    TABULAR_MODEL: Type[TabularModel]
    TILE_LEARNER: Optional[str] = None
    # It reads the raw rows, so it gets no text encoder; the model it wraps preprocesses TILE's input as usual.
    USE_TEXT_EMBEDDINGS = False
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TARGET_ENCODER = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        model = cls.TABULAR_MODEL
        cls.MODEL_NAME, cls.SHORT_NAME = f"TILE-{model.MODEL_NAME}", f"tile{model.SHORT_NAME}"
        cls.NEEDS_GPU, cls.ENV, cls.REQUIRED_ENV = model.NEEDS_GPU, model.ENV, model.REQUIRED_ENV

    def __init__(self, problem_type: SupervisedTask, device: torch.device, **kwargs):
        if importlib.util.find_spec("tile") is None:
            raise ImportError("TILE needs its package: pip install --no-deps -r requirements-tile.in")
        super().__init__(problem_type=problem_type, device=device, **kwargs)
        self.tabular_model = self.TABULAR_MODEL(problem_type=problem_type, device=device, **kwargs)
        self.rows = TileRows(image_column=self.image_column, device=device)

    def initialize_model(self) -> None:
        return None

    def fit(self, x: DataFrame, y: Series):
        from tile.features import TILEFeatures
        from tile.learners import LEARNERS, Learner
        if self.image_column:
            raise ValueError("TILE runs on text datasets only")
        learner = LEARNERS[self.TILE_LEARNER] if self.TILE_LEARNER else Learner()
        self.rows.string_columns = tile_string_columns(x, self.image_column)
        if self.dataset is not None:
            self.rows.cache = cached_embeddings(Interface.TILE, self.dataset)
            print(f"Reading the TILE embeddings of {self.rows.string_columns} from the cache")
        self.features = TILEFeatures(window=learner.window, native_categories=learner.native_categories)
        table = self._see(x)
        task = "classification" if self.is_cls else "regression"
        z = self.features.fit_transform(table, y.to_numpy(), self._embeddings(table), task)
        self.tabular_model.fit(self._frame(z, table), y)

    def predict(self, x: DataFrame) -> np.ndarray:
        return self.tabular_model.predict(self._transform(x))

    def score_all_metrics(self, X: DataFrame, y: Series) -> Dict:
        return self.tabular_model.score_all_metrics(self._transform(X), y)

    def _transform(self, x: DataFrame) -> DataFrame:
        table = self._see(x)
        return self._frame(self.features.transform(table, self._embeddings(table)), table)

    def _see(self, x: DataFrame) -> DataFrame:
        self.rows.add(x)
        return tile_table(x, self.image_column)

    def _embeddings(self, table: DataFrame) -> Dict[str, np.ndarray]:
        cells = self.rows.cells(table.index)
        return {col: block.astype(np.float32) for col, block in zip(self.rows.string_columns, cells)}

    def _frame(self, z: np.ndarray, table: DataFrame) -> DataFrame:
        """TILEFeatures' array as a table: the features as floats, named after their columns, then the categories."""
        features = self.features
        names = [f"{col}_txt_pca_{i}" for col in features.string_columns_
                 for i in range(features.reducers_[col][-1].n_components_)]
        if features.other_columns_:
            names += [str(name) for name in features.vectorizer_.get_feature_names_out()]
        width = features.target_encoder_.n_classes_ if features.target_encoder_.n_classes_ > 2 else 1
        names += [f"{col}_te_{k}" for col in features.target_encoder_.columns_ for k in range(width)]
        assert len(names) == features.n_base_ == len(set(names)), f"{features.n_base_} features named {names}"
        frame = DataFrame(z[:, :features.n_base_].astype(float), index=table.index, columns=names)
        for col, values in zip(features.categories_, z[:, features.n_base_:].T):
            frame[col] = pd.Categorical(values)
        return frame
