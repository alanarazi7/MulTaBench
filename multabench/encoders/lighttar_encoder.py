"""LightTAR (https://github.com/eilamshapira/LightTAR) on frozen E5 or DINO: each column's last and middle layers
projected, on the training rows and their target, to LightTAR's default 30 PCA and 5 supervised features. Gated
LightTAR lets an inner-fold LightGBM vote choose, once per table, between that and frozen PCA-30 of the last layer."""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from lighttar import GatedLightTAR, LightTAR, LightTARProjection
from lighttar.encoder import resolve_layers
from pandas import DataFrame, Series

from multabench.embeddings.hub import CachedEmbeddings
from multabench.encoders.abstract_encoder import EmbeddingEncoder
from multabench.encoders.column_encoders import ColumnEncoder, column_caches, column_texts
from multabench.encoders.constants import LIGHTTAR_GATED, LIGHTTAR_LAST_LAYER
from multabench.encoders.dino.dino_encoder import DinoColumnEncoder, get_image_encoder, image_urls_to_embeddings
from multabench.encoders.e5.e5_encoder import E5ColumnEncoder
from multabench.encoders.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5


def middle_layer(num_hidden_layers: int) -> int:
    """The index into the encoder's hidden states of LightTAR's default middle layer."""
    return resolve_layers(["middle"], num_hidden_layers)[0]


class LightTARProjected:
    """A column encoder whose (rows, [last, middle], dim) embeddings LightTAR projects; needs y and is_cls."""
    y: Series
    is_cls: bool
    kind = "lighttar"

    def fit(self, embeddings: np.ndarray):
        task = "classification" if self.is_cls else "regression"
        self.projection = LightTARProjection(task=task).fit(embeddings, np.asarray(self.y))
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return self.projection.transform(embeddings)

    @property
    def features(self) -> List[str]:
        width = self.projection.n_components_ if self.kind == "frozen" else self.projection.components_.shape[1]
        return [f"{self.col}_{self.SUFFIX}_{self.kind}_{i}" for i in range(width)]


class LightTARColumnEncoder(LightTARProjected, E5ColumnEncoder):
    """A text column's E5 states, read from the two caches or encoded, then LightTAR."""

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


class LightTARImageColumnEncoder(LightTARProjected, DinoColumnEncoder):
    """An image column's DINO CLS tokens before the final norm, read from its cache or encoded, then LightTAR."""

    def __init__(self, col: str, model, processor, image_folder: str, y: Series, is_cls: bool,
                 cache: Optional[CachedEmbeddings] = None):
        super().__init__(col, model, processor, image_folder, cache=cache)
        self.y, self.is_cls = y, is_cls

    def encode(self, values: Series) -> np.ndarray:
        layers = [-1, middle_layer(self.model.config.num_hidden_layers)]
        return image_urls_to_embeddings(values, image_folder=self.image_folder, processor=self.processor,
                                        model=self.model, layers=layers)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return super().transform(embeddings).astype(np.float32)


class RunTask(LightTAR):
    """LightTAR on the run's task, is_cls, rather than one guessed from y."""
    is_cls: bool

    def _setup(self, X, y):
        df, y = super()._setup(X, y)
        self.is_classification_ = self.is_cls
        return df, y


class RunTaskGatedLightTAR(GatedLightTAR, RunTask):
    pass


class LightTARTableEncoder(EmbeddingEncoder):
    """Plain LightTAR fits each column alone; the gated one fits lighttar's GatedLightTAR on the whole table, whose
    vote picks LightTAR or frozen PCA-30 for all the columns, from the plain one's caches."""

    @property
    def cache_name(self):
        return LIGHTTAR_GATED.get(self.name, self.name)

    def caches(self, columns: List[str]) -> Dict[str, CachedEmbeddings]:
        return column_caches(self.cache_name, self.dataset, columns)

    def fit(self, x: DataFrame, columns: List[str], y: Optional[Series] = None,
            is_cls: bool = True) -> Tuple[Dict[str, ColumnEncoder], Dict[str, np.ndarray]]:
        if self.name not in LIGHTTAR_GATED or not columns:
            return super().fit(x, columns, y=y, is_cls=is_cls)
        encoders = self.column_encoders(x, columns, y=y, is_cls=is_cls)
        states = {encoder.col: encoder.embed(x) for encoder in encoders}
        gate = RunTaskGatedLightTAR(columns=columns, encoder=None, n_jobs=len(os.sched_getaffinity(0)))
        gate.is_cls = is_cls
        gate.fit(x, np.asarray(y), layer_states=states)
        for encoder in encoders:
            encoder.projection, encoder.kind = gate.models_[encoder.col], gate.gate_decision_
        return {encoder.col: encoder for encoder in encoders}, states


class LightTAREncoder(LightTARTableEncoder):
    """LightTAR on the text columns, with the last layer from the frozen E5's cache and the middle from its own."""

    def column_encoders(self, x: DataFrame, columns: List[str], y: Optional[Series],
                        is_cls: bool) -> List[LightTARColumnEncoder]:
        last = column_caches(LIGHTTAR_LAST_LAYER[self.cache_name], self.dataset, columns)
        middle = self.caches(columns)
        model, tokenizer = None, None
        if not (set(last) & set(middle)).issuperset(columns):
            model, tokenizer = get_vanilla_e5(self.device, model_name=self.spec.encoder_name)
        return [LightTARColumnEncoder(col, model, tokenizer, self.device, y=y, is_cls=is_cls,
                                      caches=(last[col], middle[col]) if col in last and col in middle else None)
                for col in columns]


class LightTARImageEncoder(LightTARTableEncoder):
    """LightTAR on the image columns, with both of DINO's layers from its own cache."""

    def column_encoders(self, x: DataFrame, columns: List[str], y: Optional[Series],
                        is_cls: bool) -> List[LightTARImageColumnEncoder]:
        if self.image_folder is None:
            raise ValueError(f"Image columns {columns} need an image_folder")
        caches = self.caches(columns)
        model, processor = None, None
        if not set(caches).issuperset(columns):
            model, processor = get_image_encoder(model_name=self.spec.encoder_name)
            model.to(self.device)
        return [LightTARImageColumnEncoder(col, model, processor, self.image_folder, y=y, is_cls=is_cls,
                                           cache=caches.get(col)) for col in columns]
