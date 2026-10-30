"""Cached embeddings on the Hugging Face Hub, one dataset repo per encoder.

A frozen encoder doesn't depend on the split, so each distinct input of a dataset is embedded once, under <dataset>/. A
tuned (-tar) encoder is fine-tuned on each fold's training rows alone, so it is cached per fold, under
<dataset>/<size>/fold_<k>/, for that fold's train and test rows, with its LoRA adapter. Each directory has
embeddings.npz, which maps every input, exactly as the encoder got it, to its embedding, and meta.json, with the timing
and the hardware it ran on. A run looks its own inputs up, so a cached embedding is never paired with a different text
or image, and a changed dataset raises instead. TILE embeds a text cell in the context of its row, so its input is the
serialized row, and its embedding all the row's text cells, (features, hidden) in float16.
"""
import json
import os
from dataclasses import dataclass
from os.path import join
from functools import lru_cache
from typing import List, Optional, Sequence

import numpy as np
from huggingface_hub import snapshot_download
from pandas import DataFrame

from multabench.baselines.preprocessing.feature_types import transform_feature_types
from multabench.benchmark.splits import Size
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.datasets.hub import HF_ORG, local_repo_dir
from multabench.datasets.revisions import revision
from multabench.dino.constants import IMAGE_ENCODERS, ImageEncoder
from multabench.e5.constants import TEXT_ENCODERS, TextEncoder, format_e5_passage
from multabench.utils.encoders import Encoder

EMBEDDINGS_NPZ = "embeddings.npz"
META_JSON = "meta.json"
# numpy pads a string array to its longest string, so the inputs are stored as one UTF-8 buffer and its offsets.
INPUTS_UTF8 = "inputs_utf8"
INPUTS_OFFSETS = "inputs_offsets"
EMBEDDINGS = "embeddings"

CachedEncoder = TextEncoder | ImageEncoder
CACHED_ENCODERS = (TextEncoder.E5_SMALL, ImageEncoder.DINO_SMALL, TextEncoder.E5_SMALL_TAR, ImageEncoder.DINO_SMALL_TAR,
                   TextEncoder.E5_LARGE, ImageEncoder.DINO_LARGE, TextEncoder.E5_LARGE_TAR, ImageEncoder.DINO_LARGE_TAR,
                   TextEncoder.TILE)
# TILE's caches add up to about 54 GB, so they are built, fetched and prefetched only when asked for.
DEFAULT_ENCODERS = tuple(e for e in CACHED_ENCODERS if e != TextEncoder.TILE)


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


def cached_encoder(model_name: str, tuned: bool) -> Optional[CachedEncoder]:
    for encoder in CACHED_ENCODERS:
        if encoder_spec(encoder) == (model_name, tuned):
            return encoder
    return None


def embeds_dataset(encoder: CachedEncoder, dataset_id: MulTaBenchDatasetID) -> bool:
    """Whether the benchmark's runs use the encoder on the dataset; a tuned text encoder only runs on text datasets."""
    modality = dataset_modality(dataset_id)
    if isinstance(encoder, ImageEncoder):
        return modality.has_images
    return modality == Modality.TEXT if is_tuned(encoder) else modality.has_text


def embeddings_dir(dataset_id: MulTaBenchDatasetID, size: Optional[Size] = None, fold: Optional[int] = None) -> str:
    """<dataset> for a frozen encoder, <dataset>/<size>/fold_<k> for a tuned one."""
    if fold is None:
        return dataset_id.value
    return join(dataset_id.value, str(Size(size)), f"fold_{fold}")


@dataclass
class CachedEmbeddings:
    inputs: List[str]
    embeddings: np.ndarray
    meta: dict

    def __post_init__(self):
        self._row = {text: i for i, text in enumerate(self.inputs)}

    def rows(self, inputs: Sequence[str], feature: str) -> List[int]:
        missing = [text for text in inputs if text not in self._row]
        if missing:
            raise KeyError(f"{len(missing)} of {len(inputs)} inputs of {feature!r} have no cached embedding, "
                           f"e.g. {missing[0][:100]!r}")
        return [self._row[text] for text in inputs]

    def lookup(self, inputs: Sequence[str], feature: str) -> np.ndarray:
        return self.embeddings[self.rows(inputs, feature)]


def tile_cells(cache: CachedEmbeddings, texts: Sequence[str], columns: List[str]) -> np.ndarray:
    """(n_columns, n_rows, hidden): the cached TILE cells of these columns, for these serialized rows."""
    cached = cache.meta["features"]
    if missing := [col for col in columns if col not in cached]:
        raise ValueError(f"The cached TILE embeddings have no {missing}, only {cached}")
    rows = cache.rows(texts, feature="rows")
    return cache.embeddings[np.ix_(rows, [cached.index(col) for col in columns])].transpose(1, 0, 2)


def input_values(x: DataFrame, encoder: CachedEncoder, col: str) -> List[str]:
    """A feature's value per row as the encoder gets it: text after a run's missing-text fill, or the image file name."""
    if isinstance(encoder, TextEncoder):
        return transform_feature_types(x[[col]], numerical_features=set(), image_features=set())[col].tolist()
    return x[col].astype(str).tolist()


def encoder_inputs(encoder: CachedEncoder, col: str, values: Sequence[str]) -> List[str]:
    """The cache's keys: e5's "passage: <column>: <value>", or the image file name."""
    if isinstance(encoder, TextEncoder):
        return e5_passages(col, values)
    return list(values)


def e5_passages(col: str, texts: Sequence[str]) -> List[str]:
    return [format_e5_passage(str(col), col_val=text) for text in texts]


def write_embeddings(path: str, inputs: List[str], embeddings: np.ndarray, meta: dict, dtype=np.float32):
    """meta.json is written last, so its presence marks a complete directory."""
    os.makedirs(path, exist_ok=True)
    assert len(inputs) == len(embeddings) == len(set(inputs)), f"{len(inputs)} inputs for {len(embeddings)} embeddings"
    encoded = [text.encode() for text in inputs]
    np.savez_compressed(join(path, EMBEDDINGS_NPZ), **{
        INPUTS_UTF8: np.frombuffer(b"".join(encoded), dtype=np.uint8),
        INPUTS_OFFSETS: np.cumsum([0] + [len(b) for b in encoded], dtype=np.int64),
        EMBEDDINGS: np.asarray(embeddings, dtype=dtype)})
    with open(join(path, META_JSON), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def read_embeddings(path: str) -> CachedEmbeddings:
    with np.load(join(path, EMBEDDINGS_NPZ)) as npz:
        buffer, offsets, embeddings = npz[INPUTS_UTF8].tobytes(), npz[INPUTS_OFFSETS], npz[EMBEDDINGS]
    inputs = [buffer[start:end].decode() for start, end in zip(offsets[:-1], offsets[1:])]
    with open(join(path, META_JSON)) as f:
        meta = json.load(f)
    return CachedEmbeddings(inputs=inputs, embeddings=embeddings, meta=meta)


def download_embeddings(encoder: CachedEncoder, dataset_id: MulTaBenchDatasetID, size: Optional[Size] = None,
                        fold: Optional[int] = None, data_dir: Optional[str] = None,
                        files: Sequence[str] = (EMBEDDINGS_NPZ, META_JSON)) -> str:
    """The directory of the dataset's cached embeddings, the fold's for a tuned encoder; under data_dir if given."""
    if is_tuned(encoder) != (fold is not None):
        raise ValueError(f"{encoder} embeddings are cached {'per fold' if is_tuned(encoder) else 'once per dataset'}")
    path_in_repo = embeddings_dir(dataset_id, size=size, fold=fold)
    repo_id = embeddings_repo_id(encoder)
    if data_dir is not None:
        return join(local_repo_dir(data_dir, repo_id, [f"{path_in_repo}/{f}" for f in files]), path_in_repo)
    dir_path = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision(repo_id),
                                 allow_patterns=[f"{path_in_repo}/{f}" for f in files])
    path = join(dir_path, path_in_repo)
    missing = [f for f in files if not os.path.exists(join(path, f))]
    if missing:
        raise FileNotFoundError(f"{repo_id} has no {path_in_repo}/{missing[0]}")
    return path


# Room for a run's text and image embeddings: a few hundred MB each, up to 5.6 GB for TILE's largest.
@lru_cache(maxsize=2)
def cached_embeddings(encoder: CachedEncoder, dataset_id: MulTaBenchDatasetID, size: Optional[Size] = None,
                      fold: Optional[int] = None, data_dir: Optional[str] = None) -> CachedEmbeddings:
    return read_embeddings(download_embeddings(encoder, dataset_id, size=size, fold=fold, data_dir=data_dir))


def reads_cache(encoder: Encoder) -> bool:
    """Whether runs read this encoder's embeddings from the cache, which must then have them, not embed live."""
    return cached_encoder(encoder.encoder_name, encoder.tune_encoder) is not None
