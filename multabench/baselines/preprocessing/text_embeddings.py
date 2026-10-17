import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppresses warning, avoids deadlock
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from pandas import DataFrame, Series

from multabench.baselines.preprocessing.column_encoders import (PCA_COMPONENTS, ColumnEncoder, column_caches,
                                                                fit_column_encoders)
from multabench.constants import SEED
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.e5.constants import E5_LORA, E5_SMALL_V2, TF_IDF
from multabench.e5.e5_finetune import encode_texts_with_e5, finetune_e5, get_vanilla_e5
from multabench.embeddings.hub import CachedEmbeddings, e5_passages
from multabench.utils.timing import embedding_step


def _texts(values: Series) -> List[str]:
    return values.astype(str).fillna("").tolist()


class E5ColumnEncoder(ColumnEncoder):
    """E5 on "passage: <column>: <value>", vanilla or tuned, shared by all the text columns."""
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


def fit_text_encoders(x: DataFrame, text_features: Set[str], device: torch.device, y: Optional[Series] = None,
                      tune_e5: bool = False, e5_train_kwargs: Optional[Dict[str, Any]] = None, is_cls: bool = True,
                      e5_model_name: str = E5_SMALL_V2, dataset: Optional[MulTaBenchDatasetID] = None
                      ) -> Tuple[Dict[str, ColumnEncoder], Dict[str, np.ndarray]]:
    """One encoder per text column, with the training rows' embeddings. Tuning fine-tunes one E5 for all the columns on
    the training rows (y); otherwise columns with cached embeddings on the Hub are read from there."""
    columns = sorted(text_features)
    if not columns:
        return {}, {}
    if e5_model_name == TF_IDF:
        return fit_column_encoders(x, [TfidfColumnEncoder(col, x[col]) for col in columns])
    tune = tune_e5 and y is not None
    caches = {} if tune else column_caches(e5_model_name, dataset)
    model, tokenizer = None, None
    if tune:
        model, tokenizer, _ = finetune_e5(x[columns], y, columns, is_cls=is_cls, device=device,
                                          model_name=e5_model_name, args=replace(E5_LORA, **(e5_train_kwargs or {})))
    elif not set(caches).issuperset(columns):
        model, tokenizer = get_vanilla_e5(device, model_name=e5_model_name)
    return fit_column_encoders(x, [E5ColumnEncoder(col, model, tokenizer, device, cache=caches.get(col))
                                   for col in columns])
