"""TILE's cell embeddings, from its package (https://github.com/soda-inria/tile): each row serialized as one text, "The
<column> is <value>. ...", read by an LLM cut after one of its middle layers, and the hidden states of each cell's
tokens mean-pooled. A cell's embedding sees the cells before it in the row, so it depends on the whole row."""
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np
import torch
from pandas import DataFrame, Index

from multabench.embeddings.hub import CachedEmbeddings, tile_cells
from multabench.interfaces.tile.constants import LLAMA_3_1_8B_REVISION
from multabench.utils.timing import embedding_step

# Not in upstream TILE: a longer value is cut, as one Zomato row, ~138k tokens, would not fit on an H100.
MAX_CELL_CHARS = 32_768

_TILE_LLM_CACHE: Dict[str, Tuple[Any, Any]] = {}


def tile_table(x: DataFrame, image_column: Optional[str]) -> DataFrame:
    """The rows as TILE reads them: without the image column, whose file names carry no text, and with the columns
    MulTaBench stores as categories as objects, so TILE takes them as strings like the text columns."""
    x = x.drop(columns=[image_column], errors="ignore")
    return x.astype({col: object for col in x.select_dtypes(include=["category", "string"]).columns})


def tile_string_columns(x: DataFrame, image_column: Optional[str]) -> List[str]:
    """The columns TILE embeds: every string column but the dates."""
    from tile.features import string_columns
    return string_columns(tile_table(x, image_column))


@embedding_step
def get_tile_llm(device: torch.device) -> Tuple[Any, Any]:
    """TILE's default LLM cut at its layer, on the run's device, as tile.llms.load loads it from tile.llms.truncate's
    copy, but at a pinned revision and without the copy on disk; only the kept blocks' weights are read."""
    if str(device) not in _TILE_LLM_CACHE:
        from tile.llms import DEFAULT_LLM, LLMS
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        llm = LLMS[DEFAULT_LLM]
        config = AutoConfig.from_pretrained(llm.repo_id, revision=LLAMA_3_1_8B_REVISION)
        config.num_hidden_layers = llm.layer
        if getattr(config, "layer_types", None) is not None:
            config.layer_types = config.layer_types[:llm.layer]
        model = AutoModel.from_pretrained(llm.repo_id, revision=LLAMA_3_1_8B_REVISION, config=config,
                                          dtype=torch.bfloat16, device_map=device)
        model.norm = torch.nn.Identity()
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(llm.repo_id, revision=LLAMA_3_1_8B_REVISION)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        _TILE_LLM_CACHE[str(device)] = model.eval(), tokenizer
    return _TILE_LLM_CACHE[str(device)]


class TileRows:
    """The rows an interface has seen, serialized as loaded, by index, and their string cells' embeddings: read from
    the dataset's cache if set, else one forward pass per row for all its string columns."""

    def __init__(self, image_column: Optional[str], device: torch.device):
        self.image_column = image_column
        self.device = device
        self.columns: List[str] = []
        self.string_columns: List[str] = []
        self.texts: Dict[Hashable, str] = {}
        self.spans: Dict[Hashable, np.ndarray] = {}
        self.cache: Optional[CachedEmbeddings] = None

    def add(self, x: DataFrame):
        """Serializes the rows before any preprocessing, as tile_table reads them."""
        from tile.embeddings import serialize
        x = tile_table(x, self.image_column)
        if not x.index.is_unique:
            raise ValueError("TILE finds a row's text by its index, which must be unique")
        if self.columns and self.columns != list(x.columns):
            raise ValueError(f"TILE saw columns {self.columns} before, now {list(x.columns)}")
        self.columns = list(x.columns)
        x = x.copy()
        for col in x.select_dtypes(include=["object", "string"]).columns:
            x[col] = x[col].map(lambda v: v[:MAX_CELL_CHARS] if isinstance(v, str) else v)
        texts, spans = serialize(x)
        self.texts.update(zip(x.index, texts))
        self.spans.update(zip(x.index, spans))

    def cells(self, index: Index) -> np.ndarray:
        """(n_string_columns, n_rows, hidden), float16."""
        texts = [self.texts[i] for i in index]
        if self.cache is not None:
            return tile_cells(self.cache, texts, self.string_columns)
        positions = [self.columns.index(col) for col in self.string_columns]
        return self._pool(texts, np.stack([self.spans[i] for i in index])[:, positions])

    @embedding_step
    def _pool(self, texts: List[str], spans: np.ndarray) -> np.ndarray:
        """(n_string_columns, n_rows, hidden), as TILE's embed_table pools them, for the string columns only."""
        from tile.embeddings import pool_cells
        model, tokenizer = get_tile_llm(self.device)
        # As embed_table: cuDNN's attention kernel fails on padded batches in some environments. Restored after, as the
        # switch is process-wide and the model's own attention would otherwise lose it.
        cudnn_sdp = torch.backends.cuda.cudnn_sdp_enabled()
        torch.backends.cuda.enable_cudnn_sdp(False)
        try:
            return pool_cells(model, tokenizer, texts, spans)
        finally:
            torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp)
