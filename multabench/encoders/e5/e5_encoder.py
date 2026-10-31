import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppresses warning, avoids deadlock
from typing import Any, List, Optional

import numpy as np
import torch
from pandas import DataFrame, Series

from multabench.embeddings.hub import CachedEmbeddings, e5_passages
from multabench.encoders.abstract_encoder import EmbeddingEncoder
from multabench.encoders.column_encoders import ColumnEncoder, column_texts
from multabench.encoders.e5.e5_finetune import encode_texts_with_e5, finetune_e5, get_vanilla_e5


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
        return encode_texts_with_e5(column_texts(values), col_name=self.col, model=self.model, tokenizer=self.tokenizer,
                                    device=self.device)

    def cache_keys(self, values: Series) -> List[str]:
        return e5_passages(self.col, column_texts(values))


class E5Encoder(EmbeddingEncoder):
    """E5 on the text columns, vanilla or, for a tuned encoder, one E5 for all of them fine-tuned on (x, y)."""

    def column_encoders(self, x: DataFrame, columns: List[str], y: Optional[Series],
                        is_cls: bool) -> List[ColumnEncoder]:
        caches = self.caches(columns)
        model, tokenizer = None, None
        if not set(caches).issuperset(columns):
            if self.spec.tune_encoder:
                model, tokenizer, _ = finetune_e5(x[columns], y, columns, is_cls=is_cls, device=self.device,
                                                  model_name=self.spec.encoder_name)
            else:
                model, tokenizer = get_vanilla_e5(self.device, model_name=self.spec.encoder_name)
        return [E5ColumnEncoder(col, model, tokenizer, self.device, cache=caches.get(col)) for col in columns]
