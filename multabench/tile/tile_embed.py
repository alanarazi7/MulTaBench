"""TILE's cell embeddings, as in https://github.com/soda-inria/tile: each row serialized as one text, "The <column> is
<value>. ...", read by Llama-3.1-8B cut after TILE_LAYERS blocks, and the hidden states of each cell's tokens
mean-pooled. A cell's embedding sees the cells before it in the row, so it depends on the whole row, not its value."""
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np
import torch
from pandas import DataFrame, Index

from multabench.tile.constants import LLAMA_3_1_8B, LLAMA_3_1_8B_REVISION, TILE_LAYERS, TILE_MAX_TOKENS
from multabench.utils.timing import embedding_step

TOKENIZE_CHUNK_ROWS = 10_000

_TILE_LLM_CACHE: Dict[str, Tuple[Any, Any]] = {}


def serialize_rows(x: DataFrame) -> Tuple[List[str], np.ndarray]:
    """Each row as one sentence per cell, in column order, with the [start, end) characters of each cell's sentence.

    Underscores in column names are read as spaces; missing and blank values as "NaN". Returns the texts and an
    (n_rows, n_columns, 2) array of spans.
    """
    names = [str(col).replace("\n", "").replace("_", " ") for col in x.columns]
    values = x.astype(str)
    missing = x.isna() | values.apply(lambda col: col.str.strip() == "")
    values = values.mask(missing, "NaN").to_numpy()
    texts = []
    spans = np.empty((len(values), len(names), 2), dtype=np.int64)
    for i, row in enumerate(values):
        cells, start = [], 0
        for j, (name, value) in enumerate(zip(names, row)):
            cell = f"The {name} is {value}."
            # A value ending in a space ends in "<value>."
            if cell.endswith(" ."):
                cell = cell[:-2] + "."
            spans[i, j] = (start, start + len(cell))
            cells.append(cell)
            start += len(cell) + 1
        texts.append(" ".join(cells))
    return texts, spans


@embedding_step
def get_tile_llm(device: torch.device) -> Tuple[Any, Any]:
    """The LLM cut after TILE_LAYERS blocks, in bfloat16, and its tokenizer; only the kept blocks' weights are read."""
    if str(device) not in _TILE_LLM_CACHE:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        config = AutoConfig.from_pretrained(LLAMA_3_1_8B, revision=LLAMA_3_1_8B_REVISION)
        config.num_hidden_layers = TILE_LAYERS
        model = AutoModel.from_pretrained(LLAMA_3_1_8B, revision=LLAMA_3_1_8B_REVISION, config=config,
                                          dtype=torch.bfloat16)
        # The paper reads the raw output of the last block kept; the final norm only follows the LLM's last block.
        model.norm = torch.nn.Identity()
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_3_1_8B, revision=LLAMA_3_1_8B_REVISION)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        # Under causal attention, right padding is never seen by a real token.
        tokenizer.padding_side = "right"
        _TILE_LLM_CACHE[str(device)] = model.to(device).eval(), tokenizer
    return _TILE_LLM_CACHE[str(device)]


@embedding_step
def pool_cells(model: Any, tokenizer: Any, texts: List[str], spans: np.ndarray,
               max_tokens: int = TILE_MAX_TOKENS) -> np.ndarray:
    """The mean of the last hidden state over each cell's tokens, as an (n_rows, n_cells, hidden) float16 array.

    A token belongs to every cell whose characters it overlaps; special and padding tokens cover none. Rows run from
    the longest, in batches of at most max_tokens tokens, padding included (a longer row goes alone).
    """
    if torch.cuda.is_available():
        # cuDNN's attention kernel fails on padded batches in some environments.
        torch.backends.cuda.enable_cudnn_sdp(False)
    n_rows, n_cells, _ = spans.shape
    lengths = np.concatenate([[len(ids) for ids in tokenizer(texts[i:i + TOKENIZE_CHUNK_ROWS])["input_ids"]]
                              for i in range(0, n_rows, TOKENIZE_CHUNK_ROWS)])
    order = np.argsort(-lengths, kind="stable")
    cells = np.empty((n_rows, n_cells, model.config.hidden_size), np.float16)
    start = 0
    while start < n_rows:
        idx = order[start:start + max(1, max_tokens // lengths[order[start]])]
        inputs = tokenizer([texts[k] for k in idx], return_tensors="pt", padding=True, return_offsets_mapping=True)
        offsets = inputs.pop("offset_mapping").to(model.device)
        with torch.inference_mode():
            hidden = model(**inputs.to(model.device)).last_hidden_state.float()
            cell = torch.as_tensor(spans[idx], device=model.device)
            tok_start, tok_end = offsets[:, None, :, 0], offsets[:, None, :, 1]
            # (batch, n_cells, seq_len)
            in_cell = ((tok_start < cell[:, :, 1:]) & (tok_end > cell[:, :, :1]) & (tok_end > tok_start)).float()
            pooled = torch.bmm(in_cell, hidden) / in_cell.sum(-1, keepdim=True)
        cells[idx] = pooled.half().cpu().numpy()
        start += len(idx)
    return cells


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
        x = x.drop(columns=[self.image_column], errors="ignore")
        if not x.index.is_unique:
            raise ValueError("TILE finds a row's text by its index, which must be unique")
        if self.columns and self.columns != list(x.columns):
            raise ValueError(f"TILE saw columns {self.columns} before, now {list(x.columns)}")
        self.columns = list(x.columns)
        texts, spans = serialize_rows(x)
        self.texts.update(zip(x.index, texts))
        self.spans.update(zip(x.index, spans))

    def embed(self, index: Index, col: str) -> np.ndarray:
        """The column's cell embeddings for these rows; the rows' other text columns are pooled in the same pass."""
        if self._index is None or not self._index.equals(index):
            self._index, self._cells = index, self._pool(index)
        return self._cells[:, self.text_columns.index(col)].astype(np.float32)

    def _pool(self, index: Index) -> np.ndarray:
        model, tokenizer = get_tile_llm(self.device)
        positions = [self.columns.index(col) for col in self.text_columns]
        spans = np.stack([self.spans[i] for i in index])[:, positions]
        return pool_cells(model, tokenizer, [self.texts[i] for i in index], spans)
