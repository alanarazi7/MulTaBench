"""LightTAR (https://github.com/eilamshapira/LightTAR) on frozen E5: each text column's last and middle layers projected,
on the training rows and their target, to LightTAR's default 30 PCA and 5 supervised features."""
from typing import List, Optional, Tuple

import numpy as np
from lighttar import LightTARProjection
from lighttar.encoder import resolve_layers
from pandas import DataFrame, Series

from multabench.embeddings.hub import CachedEmbeddings
from multabench.encoders.abstract_encoder import EmbeddingEncoder
from multabench.encoders.column_encoders import column_caches, column_texts
from multabench.encoders.constants import LIGHTTAR_LAST_LAYER
from multabench.encoders.e5.e5_encoder import E5ColumnEncoder
from multabench.encoders.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5


def middle_layer(num_hidden_layers: int) -> int:
    """The index into E5's hidden states of LightTAR's default middle layer."""
    return resolve_layers(["middle"], num_hidden_layers)[0]


class LightTARColumnEncoder(E5ColumnEncoder):
    """A text column's (rows, [last, middle], dim) E5 states, read from the two caches or encoded, then LightTAR."""

    def __init__(self, col: str, model, tokenizer, device, y: Series, is_cls: bool,
                 caches: Optional[Tuple[CachedEmbeddings, CachedEmbeddings]] = None):
        super().__init__(col, model, tokenizer, device)
        self.y, self.is_cls, self.caches = y, is_cls, caches

    def embed(self, x: DataFrame) -> np.ndarray:
        if self.caches is None:
            return self.encode(x[self.col])
        keys = self.cache_keys(x[self.col])
        return np.stack([cache.lookup(keys, feature=self.col) for cache in self.caches], axis=1)

    def encode(self, values: Series) -> np.ndarray:
        layers = [-1, middle_layer(self.model.config.num_hidden_layers)]
        return encode_texts_with_e5(column_texts(values), col_name=self.col, model=self.model, tokenizer=self.tokenizer,
                                    device=self.device, layers=layers)

    def fit(self, embeddings: np.ndarray) -> "LightTARColumnEncoder":
        task = "classification" if self.is_cls else "regression"
        self.projection = LightTARProjection(task=task).fit(embeddings, np.asarray(self.y))
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.projection.transform(embeddings)

    @property
    def features(self) -> List[str]:
        return [f"{self.col}_{self.SUFFIX}_lighttar_{i}" for i in range(self.projection.components_.shape[1])]


class LightTAREncoder(EmbeddingEncoder):
    """LightTAR on the text columns, with the last layer from the frozen E5's cache and the middle from its own."""

    def column_encoders(self, x: DataFrame, columns: List[str], y: Optional[Series],
                        is_cls: bool) -> List[LightTARColumnEncoder]:
        last = column_caches(LIGHTTAR_LAST_LAYER[self.name], self.dataset, columns)
        middle = self.caches(columns)
        model, tokenizer = None, None
        if not (set(last) & set(middle)).issuperset(columns):
            model, tokenizer = get_vanilla_e5(self.device, model_name=self.spec.encoder_name)
        return [LightTARColumnEncoder(col, model, tokenizer, self.device, y=y, is_cls=is_cls,
                                      caches=(last[col], middle[col]) if col in last and col in middle else None)
                for col in columns]
