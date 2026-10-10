"""Cached embeddings on the Hugging Face Hub, one dataset repo per encoder.

The encoders are frozen, so they don't depend on the split: every row of a dataset is embedded once, under
<dataset>/, in embeddings.npz, with the row positions in the dataset under ROWS_KEY and one (n_rows, dim) array per
feature, and meta.json, with the encoding time and the hardware it ran on.
"""
import json
import os
from os.path import join
from typing import Dict, Tuple

import numpy as np

from multabench.datasets.all_datasets import MulTaBenchDatasetID, dataset_modality
from multabench.datasets.hub import HF_ORG
from multabench.dino.constants import IMAGE_ENCODERS, ImageEncoder
from multabench.e5.constants import TEXT_ENCODERS, TextEncoder
from multabench.utils.encoders import Encoder

EMBEDDINGS_NPZ = "embeddings.npz"
META_JSON = "meta.json"
ROWS_KEY = "__rows__"

CachedEncoder = TextEncoder | ImageEncoder
CACHED_ENCODERS = (TextEncoder.E5_SMALL, ImageEncoder.DINO_SMALL)
# The upload's commit; "main" until an encoder's embeddings are on the Hub.
REVISIONS: Dict[CachedEncoder, str] = {TextEncoder.E5_SMALL: "21bb46972d46c25fe08f035667f4cabdd95bc83e",
                                       ImageEncoder.DINO_SMALL: "main"}


def parse_encoder(encoder: str) -> CachedEncoder:
    for cached in CACHED_ENCODERS:
        if cached == encoder:
            return cached
    raise ValueError(f"No cached embeddings for {encoder!r}; choose from {[str(e) for e in CACHED_ENCODERS]}")


def encoder_spec(encoder: CachedEncoder) -> Encoder:
    return TEXT_ENCODERS[encoder] if isinstance(encoder, TextEncoder) else IMAGE_ENCODERS[encoder]


def repo_name(encoder: CachedEncoder) -> str:
    return f"embeddings-{encoder}"


def embeddings_repo_id(encoder: CachedEncoder) -> str:
    return f"{HF_ORG}/{repo_name(encoder)}"


def embeds_dataset(encoder: CachedEncoder, dataset_id: MulTaBenchDatasetID) -> bool:
    modality = dataset_modality(dataset_id)
    return modality.has_images if isinstance(encoder, ImageEncoder) else modality.has_text


def embeddings_dir(dataset_id: MulTaBenchDatasetID) -> str:
    return dataset_id.value


def write_embeddings(path: str, rows: np.ndarray, embeddings: Dict[str, np.ndarray], meta: dict):
    """meta.json is written last, so its presence marks a complete directory."""
    os.makedirs(path, exist_ok=True)
    for feature, array in embeddings.items():
        assert len(array) == len(rows), f"{feature}: {len(array)} embeddings for {len(rows)} rows"
    np.savez(join(path, EMBEDDINGS_NPZ), **{ROWS_KEY: np.asarray(rows, dtype=np.int64)},
             **{feature: np.asarray(array, dtype=np.float32) for feature, array in embeddings.items()})
    with open(join(path, META_JSON), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def read_embeddings(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], dict]:
    with np.load(join(path, EMBEDDINGS_NPZ)) as npz:
        rows = npz[ROWS_KEY]
        embeddings = {feature: npz[feature] for feature in npz.files if feature != ROWS_KEY}
    with open(join(path, META_JSON)) as f:
        meta = json.load(f)
    return rows, embeddings, meta


def select_rows(rows: np.ndarray, embeddings: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    """The embeddings of the rows at positions idx of the dataset, in that order."""
    order = np.argsort(rows)
    pos = order[np.searchsorted(rows, idx, sorter=order).clip(max=len(rows) - 1)]
    missing = rows[pos] != idx
    if missing.any():
        raise KeyError(f"{missing.sum()} rows have no cached embedding, e.g. row {idx[missing][0]}")
    return {feature: array[pos] for feature, array in embeddings.items()}
