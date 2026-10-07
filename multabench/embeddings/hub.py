"""Cached embeddings on the Hugging Face Hub, one dataset repo per encoder.

A frozen encoder doesn't depend on the split, so it embeds every row of a dataset once, under <dataset>/. A
fine-tuned ("-tar") encoder is trained on one split's training rows, so each (size, fold) has its own fine-tune,
under <dataset>/<size>/fold-<k>/, embedding only that split's rows, next to its LoRA adapter.

Each directory holds embeddings.npz, with the row positions in the dataset under ROWS_KEY and one
(n_rows, dim) array per feature, and meta.json, with the encoding time and the hardware it ran on.
"""
import json
import os
from os.path import join
from typing import Dict, Optional, Tuple

import numpy as np

from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.datasets.hub import HF_ORG
from multabench.dino.constants import IMAGE_ENCODERS, ImageEncoder
from multabench.e5.constants import TEXT_ENCODERS, TextEncoder
from multabench.utils.encoders import Encoder

EMBEDDINGS_NPZ = "embeddings.npz"
META_JSON = "meta.json"
ADAPTER_DIR = "adapter"
ROWS_KEY = "__rows__"

CachedEncoder = TextEncoder | ImageEncoder
CACHED_ENCODERS = (TextEncoder.E5_SMALL, TextEncoder.E5_SMALL_TAR, ImageEncoder.DINO_SMALL, ImageEncoder.DINO_SMALL_TAR)
# Pinned to the upload's commit once the embeddings are on the Hub.
REVISIONS: Dict[CachedEncoder, str] = {encoder: "main" for encoder in CACHED_ENCODERS}


def parse_encoder(encoder: str) -> CachedEncoder:
    for cached in CACHED_ENCODERS:
        if cached == encoder:
            return cached
    raise ValueError(f"No cached embeddings for {encoder!r}; choose from {[str(e) for e in CACHED_ENCODERS]}")


def encoder_spec(encoder: CachedEncoder) -> Encoder:
    return TEXT_ENCODERS[encoder] if isinstance(encoder, TextEncoder) else IMAGE_ENCODERS[encoder]


def is_tuned(encoder: CachedEncoder) -> bool:
    return encoder_spec(encoder).tune_encoder


def repo_name(encoder: CachedEncoder) -> str:
    return f"embeddings-{encoder}"


def embeddings_repo_id(encoder: CachedEncoder) -> str:
    return f"{HF_ORG}/{repo_name(encoder)}"


def embeds_dataset(encoder: CachedEncoder, dataset_id: MulTaBenchDatasetID) -> bool:
    """Whether the benchmark runs need this encoder on this dataset; image datasets with text columns use frozen E5."""
    modality = dataset_modality(dataset_id)
    if isinstance(encoder, ImageEncoder):
        return modality.has_images
    if is_tuned(encoder):
        return modality == Modality.TEXT
    return modality.has_text


def embeddings_dir(dataset_id: MulTaBenchDatasetID, encoder: CachedEncoder, size: Optional[str] = None,
                   fold: Optional[int] = None) -> str:
    """Path inside the encoder's repo; size and fold only apply to fine-tuned encoders."""
    if not is_tuned(encoder):
        return dataset_id.value
    assert size is not None and fold is not None, f"{encoder} is fine-tuned per split, so it needs a size and a fold"
    return f"{dataset_id.value}/{size}/fold-{fold}"


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
