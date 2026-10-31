"""TILE's cell embeddings, from its package (https://github.com/soda-inria/tile): each row serialized as one text, "The
<column> is <value>. ...", read by an LLM cut after one of its middle layers, and the hidden states of each cell's
tokens mean-pooled. A cell's embedding sees the cells before it in the row, so it depends on the whole row."""
import importlib.util
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np
import torch
from pandas import DataFrame, Index, Series

from multabench.encoders.abstract_encoder import EmbeddingEncoder
from multabench.encoders.column_encoders import ColumnEncoder
from multabench.encoders.tile.constants import LLAMA_3_1_8B_REVISION
from multabench.utils.timing import embedding_step

# Not in upstream TILE: a longer value is cut, as one Zomato row, ~138k tokens, would not fit on an H100.
MAX_CELL_CHARS = 32_768

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
    """The rows a model has seen, serialized as loaded, by index, and their text cells' embeddings: one forward pass
    per row, shared by all its text columns."""

    def __init__(self, image_column: Optional[str], device: torch.device):
        self.image_column = image_column
        self.device = device
        self.columns: List[str] = []
        self.text_columns: List[str] = []
        self.texts: Dict[Hashable, str] = {}
        self.spans: Dict[Hashable, np.ndarray] = {}
        self._texts: Optional[List[str]] = None
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
        x = x.copy()
        for col in x.select_dtypes(include=["object", "string"]).columns:
            x[col] = x[col].map(lambda v: v[:MAX_CELL_CHARS] if isinstance(v, str) else v)
        texts, spans = serialize(x)
        self.texts.update(zip(x.index, texts))
        self.spans.update(zip(x.index, spans))

    def embed(self, index: Index, col: str) -> np.ndarray:
        """The column's cell embeddings for these rows; the rows' other text columns are pooled in the same pass, kept
        for as long as the next column asks for the same rows' texts."""
        texts = [self.texts[i] for i in index]
        if texts != self._texts:
            positions = [self.columns.index(col) for col in self.text_columns]
            self._texts, self._cells = texts, self._pool(texts, np.stack([self.spans[i] for i in index])[:, positions])
        return self._cells[self.text_columns.index(col)].astype(np.float32)

    @embedding_step
    def _pool(self, texts: List[str], spans: np.ndarray) -> np.ndarray:
        """(n_text_columns, n_rows, hidden), as TILE's embed_table pools them, for the text columns only."""
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


class TileColumnEncoder(ColumnEncoder):
    """A text column's cells as TILE embeds them, in the context of their whole row; standardized, then PCA."""
    SUFFIX = "txt"

    def __init__(self, col: str, rows: TileRows):
        super().__init__(col)
        self.rows = rows

    def embed(self, x: DataFrame) -> np.ndarray:
        return self.rows.embed(x.index, self.col)


class TileEncoder(EmbeddingEncoder):
    """TILE on the text columns: it reads the rows as loaded, which see_rows records at fit and predict."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if importlib.util.find_spec("tile") is None:
            raise ImportError("--text_encoder tile needs TILE's package: pip install --no-deps -r requirements-tile.in")
        self.rows = TileRows(image_column=self.image_column, device=self.device)

    def see_rows(self, x: DataFrame):
        self.rows.add(x)

    def column_encoders(self, x: DataFrame, columns: List[str], y: Optional[Series],
                        is_cls: bool) -> List[ColumnEncoder]:
        if not self.rows.texts:
            raise ValueError("TILE embeds whole rows: see_rows must be given the rows first")
        self.rows.text_columns = columns
        return [TileColumnEncoder(col, self.rows) for col in columns]
