from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pandas import DataFrame, Series

from multabench.benchmark.splits import Size
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.embeddings.hub import CACHED_ENCODERS, CachedEmbeddings, CachedEncoder, encoder_spec
from multabench.encoders.column_encoders import ColumnEncoder, column_caches, fit_column_encoders


class EmbeddingEncoder:
    """A run's encoder of its text or image columns, built per run like a TabularModel: fit gives one ColumnEncoder
    per column, with the training rows' embeddings, read from the cache when the encoder has one."""

    def __init__(self, name: CachedEncoder, device: torch.device, dataset: Optional[MulTaBenchDatasetID] = None,
                 size: Optional[Size] = None, fold: Optional[int] = None, image_folder: Optional[str] = None):
        self.name = name
        self.spec = encoder_spec(name)
        self.device = device
        self.dataset = dataset
        # A tuned encoder's cached embeddings are per fold.
        self.size = size
        self.fold = fold
        self.image_folder = image_folder

    def fit(self, x: DataFrame, columns: List[str], y: Optional[Series] = None,
            is_cls: bool = True) -> Tuple[Dict[str, ColumnEncoder], Dict[str, np.ndarray]]:
        if not columns:
            return {}, {}
        return fit_column_encoders(x, self.column_encoders(x, columns, y=y, is_cls=is_cls))

    def column_encoders(self, x: DataFrame, columns: List[str], y: Optional[Series],
                        is_cls: bool) -> List[ColumnEncoder]:
        raise NotImplementedError

    def caches(self, columns: List[str]) -> Dict[str, CachedEmbeddings]:
        if self.name not in CACHED_ENCODERS:
            return {}
        return column_caches(self.name, self.dataset, columns, size=self.size, fold=self.fold)
