from typing import List, Optional

import numpy as np
from pandas import DataFrame, Series

from multabench.constants import SEED
from multabench.encoders.abstract_encoder import EmbeddingEncoder
from multabench.encoders.column_encoders import PCA_COMPONENTS, ColumnEncoder, column_texts
from multabench.utils.timing import embedding_step


class TfidfColumnEncoder(ColumnEncoder):
    """skrub's StringEncoder, TF-IDF then TruncatedSVD to PCA_COMPONENTS, fitted on the training rows' texts."""
    SUFFIX = "txt"

    @embedding_step
    def __init__(self, col: str, texts: Series):
        from skrub import StringEncoder
        super().__init__(col)
        string_encoder = StringEncoder(n_components=PCA_COMPONENTS, random_state=SEED)
        self.string_encoder = string_encoder.fit(Series(column_texts(texts)))

    @embedding_step
    def encode(self, values: Series) -> np.ndarray:
        return np.asarray(self.string_encoder.transform(Series(column_texts(values), dtype=str)))

    def fit(self, embeddings: np.ndarray) -> "TfidfColumnEncoder":
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings


class TfidfEncoder(EmbeddingEncoder):
    """TF-IDF on each text column, fitted on the training rows' texts; never cached."""

    def column_encoders(self, x: DataFrame, columns: List[str], y: Optional[Series],
                        is_cls: bool) -> List[ColumnEncoder]:
        return [TfidfColumnEncoder(col, x[col]) for col in columns]
