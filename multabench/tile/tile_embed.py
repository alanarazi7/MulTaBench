"""TILE's cell embeddings, from its package (https://github.com/soda-inria/tile): each row serialized as one text, "The
<column> is <value>. ...", read by an LLM cut after one of its middle layers, and the hidden states of each cell's
tokens mean-pooled. A cell's embedding sees the cells before it in the row, so it depends on the whole row."""
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np
import torch
from pandas import DataFrame, Index

from multabench.tile.constants import LLAMA_3_1_8B_REVISION
from multabench.utils.timing import embedding_step

_TILE_LLM_CACHE: Dict[str, Tuple[Any, Any]] = {}


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
        model = AutoModel.from_pretrained(llm.repo_id, revision=LLAMA_3_1_8B_REVISION, config=config,
                                          dtype=torch.bfloat16)
        model.norm = torch.nn.Identity()
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(llm.repo_id, revision=LLAMA_3_1_8B_REVISION)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        _TILE_LLM_CACHE[str(device)] = model.to(device).eval(), tokenizer
    return _TILE_LLM_CACHE[str(device)]


class TileRows:
    """The rows a model has seen, serialized as loaded, by index, and their text cells' embeddings: one forward pass
    per row, shared by all its text columns."""

    def __init__(self, image_column: Optional[str], device: torch.device):
        self.image_column = image_column
        self.device = device
        self.columns: List[str] = []
        self.text_columns: List[str] = []
        self.texts: Dict[Hashable, str] = {}
        self.spans: Dict[Hashable, np.ndarray] = {}
        self._index: Optional[Index] = None
        self._cells: Optional[np.ndarray] = None

    def add(self, x: DataFrame):
        """Serializes the rows before any preprocessing, without the image column, whose file names carry no text."""
        from tile.embeddings import serialize
        x = x.drop(columns=[self.image_column], errors="ignore")
        if not x.index.is_unique:
            raise ValueError("TILE finds a row's text by its index, which must be unique")
        if self.columns and self.columns != list(x.columns):
            raise ValueError(f"TILE saw columns {self.columns} before, now {list(x.columns)}")
        self.columns = list(x.columns)
        texts, spans = serialize(x)
        self.texts.update(zip(x.index, texts))
        self.spans.update(zip(x.index, spans))

    def embed(self, index: Index, col: str) -> np.ndarray:
        """The column's cell embeddings for these rows; the rows' other text columns are pooled in the same pass."""
        if self._index is None or not self._index.equals(index):
            self._index, self._cells = index, self._pool(index)
        return self._cells[self.text_columns.index(col)].astype(np.float32)

    @embedding_step
    def _pool(self, index: Index) -> np.ndarray:
        """(n_text_columns, n_rows, hidden), as TILE's embed_table pools them, for the text columns only."""
        from tile.embeddings import pool_cells
        model, tokenizer = get_tile_llm(self.device)
        if torch.cuda.is_available():
            # As embed_table: cuDNN's attention kernel fails on padded batches in some environments.
            torch.backends.cuda.enable_cudnn_sdp(False)
        positions = [self.columns.index(col) for col in self.text_columns]
        spans = np.stack([self.spans[i] for i in index])[:, positions]
        return pool_cells(model, tokenizer, [self.texts[i] for i in index], spans)
