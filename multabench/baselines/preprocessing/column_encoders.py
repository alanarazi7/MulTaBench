"""Embedded columns, text or image: each value embedded, read from a cache or encoded, then the column's embeddings
reduced to PCA_COMPONENTS features named <column>_<SUFFIX>_pca_<i>, fitted on the training rows."""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from multabench.benchmark.splits import Size
from multabench.constants import SEED
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.embeddings.hub import CachedEmbeddings, CachedEncoder, cached_embeddings, is_tuned

PCA_COMPONENTS = 30


class ColumnEncoder:
    SUFFIX: str
    # Image embeddings are standardized before PCA, text embeddings aren't.
    SCALE: bool = False

    def __init__(self, col: str, cache: Optional[CachedEmbeddings] = None):
        self.col = col
        self.cache = cache

    def embed(self, x: DataFrame) -> np.ndarray:
        values = x[self.col]
        if self.cache is None:
            return self.encode(values)
        return self.cache.lookup(self.cache_keys(values), feature=self.col)

    def encode(self, values: Series) -> np.ndarray:
        raise NotImplementedError

    def cache_keys(self, values: Series) -> List[str]:
        raise NotImplementedError

    def fit(self, embeddings: np.ndarray) -> "ColumnEncoder":
        scaler = [StandardScaler()] if self.SCALE else []
        self.reduction = make_pipeline(*scaler, PCA(n_components=PCA_COMPONENTS, random_state=SEED)).fit(embeddings)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.reduction.transform(embeddings)

    @property
    def features(self) -> List[str]:
        return [f"{self.col}_{self.SUFFIX}_pca_{i}" for i in range(PCA_COMPONENTS)]


def fit_column_encoders(x: DataFrame, encoders: Sequence[ColumnEncoder]) -> Tuple[Dict[str, ColumnEncoder],
                                                                                   Dict[str, np.ndarray]]:
    """The fitted encoders by column, and the training rows' embeddings, which transform_columns can then reuse."""
    train_embeddings = {}
    for encoder in encoders:
        train_embeddings[encoder.col] = embeddings = encoder.embed(x)
        encoder.fit(embeddings)
    return {encoder.col: encoder for encoder in encoders}, train_embeddings


def transform_columns(x: DataFrame, encoders: Dict[str, ColumnEncoder],
                      train_embeddings: Optional[Dict[str, np.ndarray]] = None) -> DataFrame:
    """Each embedded column replaced by its reduced embeddings, appended after the other columns."""
    train_embeddings = train_embeddings or {}
    for col, encoder in encoders.items():
        embeddings = train_embeddings.get(col)
        if embeddings is None:
            embeddings = encoder.embed(x)
        assert len(embeddings) == len(x), f"{len(embeddings)} embeddings for {len(x)} rows of {col!r}"
        reduced = DataFrame(encoder.transform(embeddings), index=x.index, columns=encoder.features)
        x = pd.concat([x.drop(columns=[col]), reduced], axis=1)
    return x


def column_caches(encoder: CachedEncoder, dataset: Optional[MulTaBenchDatasetID], columns: List[str],
                  size: Optional[Size] = None, fold: Optional[int] = None) -> Dict[str, CachedEmbeddings]:
    """The cached embeddings of each column that has them. A tuned encoder's are its fold's, tuned on all the columns
    together, so they must be exactly the run's columns."""
    if dataset is None:
        return {}
    tuned = is_tuned(encoder)
    if tuned and fold is None:
        raise ValueError(f"The cached {encoder} embeddings are per fold; give the run's fold and size")
    cache = cached_embeddings(encoder, dataset, size=size if tuned else None, fold=fold if tuned else None)
    cached = cache.meta["features"]
    if tuned and sorted(cached) != sorted(columns):
        raise ValueError(f"The cached {encoder} embeddings were tuned on {cached}, but the run's columns are {columns}")
    print(f"Reading the embeddings of {cached} from the cached {encoder} embeddings")
    return {col: cache for col in cached}
