import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppresses warning, avoids deadlock
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from pandas import DataFrame, Series

from multabench.baselines.preprocessing.column_encoders import (PCA_COMPONENTS, ColumnEncoder, column_caches,
                                                                fit_column_encoders)
from multabench.benchmark.splits import Size
from multabench.constants import SEED
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.e5.constants import DEFAULT_TEXT_ENCODER, TEXT_ENCODERS, TextEncoder
from multabench.e5.e5_finetune import encode_texts_with_e5, finetune_e5, get_vanilla_e5
from multabench.embeddings.hub import CACHED_ENCODERS, CachedEmbeddings, e5_passages
from multabench.tile.tile_embed import TileRows
from multabench.utils.timing import embedding_step


def _texts(values: Series) -> List[str]:
    return values.astype(str).fillna("").tolist()


class E5ColumnEncoder(ColumnEncoder):
    """E5 on "passage: <column>: <value>", vanilla or tuned, shared by all text columns; standardized, then PCA."""
    SUFFIX = "txt"

    def __init__(self, col: str, model: Any, tokenizer: Any, device: torch.device,
                 cache: Optional[CachedEmbeddings] = None):
        super().__init__(col, cache=cache)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, values: Series) -> np.ndarray:
        return encode_texts_with_e5(_texts(values), col_name=self.col, model=self.model, tokenizer=self.tokenizer,
                                    device=self.device)

    def cache_keys(self, values: Series) -> List[str]:
        return e5_passages(self.col, _texts(values))


class TfidfColumnEncoder(ColumnEncoder):
    """skrub's StringEncoder, TF-IDF then TruncatedSVD to PCA_COMPONENTS, fitted on the training rows' texts."""
    SUFFIX = "txt"

    @embedding_step
    def __init__(self, col: str, texts: Series):
        from skrub import StringEncoder
        super().__init__(col)
        self.string_encoder = StringEncoder(n_components=PCA_COMPONENTS, random_state=SEED).fit(Series(_texts(texts)))

    @embedding_step
    def encode(self, values: Series) -> np.ndarray:
        return np.asarray(self.string_encoder.transform(Series(_texts(values), dtype=str)))

    def fit(self, embeddings: np.ndarray) -> "TfidfColumnEncoder":
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings


class TileColumnEncoder(ColumnEncoder):
    """A text column's cells as TILE embeds them, in the context of their whole row; standardized, then PCA."""
    SUFFIX = "txt"

    def __init__(self, col: str, rows: TileRows):
        super().__init__(col)
        self.rows = rows

    def embed(self, x: DataFrame) -> np.ndarray:
        return self.rows.embed(x.index, self.col)


def fit_text_encoders(x: DataFrame, text_features: Set[str], device: torch.device,
                      encoder: TextEncoder = DEFAULT_TEXT_ENCODER, y: Optional[Series] = None, is_cls: bool = True,
                      dataset: Optional[MulTaBenchDatasetID] = None, size: Optional[Size] = None,
                      fold: Optional[int] = None,
                      tile_rows: Optional[TileRows] = None) -> Tuple[Dict[str, ColumnEncoder], Dict[str, np.ndarray]]:
    """One encoder per text column, with the training rows' embeddings, read from the cache when the encoder has
    one: the fold's, for a tuned encoder. Otherwise a tuned encoder is one E5 for all the columns, tuned on (x, y).
    TILE reads the rows the model saw, as loaded, from tile_rows."""
    columns = sorted(text_features)
    if not columns:
        return {}, {}
    if encoder == TextEncoder.TFIDF:
        return fit_column_encoders(x, [TfidfColumnEncoder(col, x[col]) for col in columns])
    if encoder == TextEncoder.TILE:
        if tile_rows is None:
            raise ValueError("TILE embeds whole rows: pass the TileRows of the rows the model saw")
        tile_rows.text_columns = columns
        return fit_column_encoders(x, [TileColumnEncoder(col, tile_rows) for col in columns])
    spec = TEXT_ENCODERS[encoder]
    caches = column_caches(encoder, dataset, columns, size=size, fold=fold) if encoder in CACHED_ENCODERS else {}
    model, tokenizer = None, None
    if not set(caches).issuperset(columns):
        if spec.tune_encoder:
            model, tokenizer, _ = finetune_e5(x[columns], y, columns, is_cls=is_cls, device=device,
                                              model_name=spec.encoder_name)
        else:
            model, tokenizer = get_vanilla_e5(device, model_name=spec.encoder_name)
    return fit_column_encoders(x, [E5ColumnEncoder(col, model, tokenizer, device, cache=caches.get(col))
                                   for col in columns])
