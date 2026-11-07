"""CASE's row embeddings, from its package (https://github.com/SAP-samples/case): CaseTransformer prefills its LLM's KV
cache with up to 128 training rows and their targets, sampled with its seed, then embeds each row as the last layer's
hidden state at its last token. As it depends on the fold's training rows, it is cached per fold."""
import re
from typing import Any, List

import numpy as np
from pandas import DataFrame, Series

from multabench.embeddings.hub import CachedEmbeddings
from multabench.interfaces.case.constants import CASE_MODEL, CASE_REVISION
from multabench.utils.timing import embedding_step

# As case.core._clean_and_tokenize_series cleans each cell before tokenizing it.
_CELL_NOISE = re.compile(r"[\t\n\r]|    ")


def case_row_keys(x: DataFrame) -> List[str]:
    """Each row's cells as CASE reads them, joined by tabs, which CASE removes from the cells."""
    return ["\t".join(row.fillna("").astype(str).str.replace(_CELL_NOISE, "", regex=True)) for _, row in x.iterrows()]


def case_rows(cache: CachedEmbeddings, x: DataFrame) -> np.ndarray:
    """(n_rows, hidden), float32: the cached embeddings of these rows."""
    return cache.lookup(case_row_keys(x), feature="rows").astype(np.float32)


@embedding_step
def load_case(x_train: DataFrame, y_train: Series) -> Any:
    """CaseTransformer with its defaults, at a pinned revision, its KV cache prefilled with the fold's training rows."""
    from case import CaseTransformer
    from huggingface_hub import snapshot_download
    case = CaseTransformer(model_name=snapshot_download(CASE_MODEL, revision=CASE_REVISION))
    # What fit does before embedding the training rows, which embed_case_rows then embeds with the test rows.
    case._setup_model(x_train, y_train)
    return case


@embedding_step
def embed_case_rows(case: Any, x: DataFrame) -> np.ndarray:
    return case.embed(x)
