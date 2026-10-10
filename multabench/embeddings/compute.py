"""Compute the cached frozen embeddings exactly as a benchmark run embeds its features live, and write them to disk."""
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pandas import Series

from multabench.baselines.preprocessing.feature_types import detect_feature_types, transform_feature_types
from multabench.baselines.preprocessing.image_embeddings import get_image_encoder, image_urls_to_embeddings
from multabench.datasets.objects import MultimodalDataset
from multabench.dino.constants import ImageEncoder
from multabench.e5.constants import TextEncoder
from multabench.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5
from multabench.embeddings.hub import CachedEncoder, encoder_spec, write_embeddings
from multabench.utils.hardware import EMBEDDING_HARDWARE, get_hardware_dict, hardware_mismatches
from multabench.utils.logging import get_current_commit_hash
from multabench.utils.timing import pop_embedding_seconds

# Every row is embedded, up to ~100K images, which don't fit in memory at once.
IMAGE_CHUNK_ROWS = 4096


def compute_embeddings(dataset: MultimodalDataset, encoder: CachedEncoder, path: str, device: torch.device):
    features = _features(dataset, encoder)
    rows = np.arange(len(dataset.y))
    hardware = {**get_hardware_dict(device), "expected_hardware": asdict(EMBEDDING_HARDWARE),
                "official_hardware_mismatches": hardware_mismatches(device, EMBEDDING_HARDWARE)}
    print(f"Embedding {len(features)} of {dataset.x.shape[1]} columns, {len(rows)} rows, with {encoder} on {hardware}")
    pop_embedding_seconds()
    model, processor = _load_frozen(encoder, device) if features else (None, None)
    load_seconds = pop_embedding_seconds()
    embeddings, seconds = _encode(dataset, encoder, features, rows, model, processor, device)
    total_seconds = load_seconds + sum(seconds.values())
    meta: Dict[str, Any] = {"dataset": dataset.dataset_id.name, "encoder": str(encoder),
                            "model": encoder_spec(encoder).encoder_name, "n_rows": len(rows),
                            "n_columns": dataset.x.shape[1], "features": features, "load_seconds": load_seconds,
                            "encode_seconds": seconds, "total_seconds": total_seconds, "git": get_current_commit_hash(),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), **hardware}
    write_embeddings(path, rows=rows, embeddings=embeddings, meta=meta)
    print(f"Embedded {dataset.dataset_id.name} with {encoder} in {total_seconds:.0f}s "
          f"({load_seconds:.0f}s loading the model)")


def _features(dataset: MultimodalDataset, encoder: CachedEncoder) -> List[str]:
    if isinstance(encoder, TextEncoder):
        return sorted(detect_feature_types(dataset.x, image_column=dataset.image_column).text_features)
    return [dataset.image_column] if dataset.image_column in dataset.x.columns else []


def _load_frozen(encoder: CachedEncoder, device: torch.device) -> Tuple[Any, Any]:
    model_name = encoder_spec(encoder).encoder_name
    if isinstance(encoder, TextEncoder):
        return get_vanilla_e5(device, model_name=model_name)
    model, processor = get_image_encoder(model_name=model_name)
    return model.to(device), processor


def _encode(dataset: MultimodalDataset, encoder: CachedEncoder, features: List[str], rows: np.ndarray, model: Any,
            processor: Any, device: torch.device) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    embeddings, seconds = {}, {}
    x = dataset.x.iloc[rows]
    for col in features:
        if isinstance(encoder, TextEncoder):
            embeddings[col] = _embed_texts(x[col], model=model, tokenizer=processor, device=device)
        elif isinstance(encoder, ImageEncoder):
            embeddings[col] = _embed_images_in_chunks(x[col], image_folder=dataset.image_folder, model=model,
                                                      processor=processor)
        else:
            raise TypeError(f"Unknown encoder {encoder!r}")
        seconds[col] = pop_embedding_seconds()
        print(f"Embedded {col}: {embeddings[col].shape} in {seconds[col]:.0f}s")
    return embeddings, seconds


def _embed_texts(texts: Series, model: Any, tokenizer: Any, device: torch.device) -> np.ndarray:
    # A run fills missing text before embedding it, so the cache embeds the same strings.
    filled = transform_feature_types(texts.to_frame(), numerical_features=set(), image_features=set())[texts.name]
    return encode_texts_with_e5(texts=filled.tolist(), col_name=str(texts.name), model=model, tokenizer=tokenizer,
                                device=device)


def _embed_images_in_chunks(urls: Series, image_folder: str, model: Any, processor: Any) -> np.ndarray:
    chunks = []
    for start in range(0, len(urls), IMAGE_CHUNK_ROWS):
        chunk = urls.iloc[start:start + IMAGE_CHUNK_ROWS]
        chunks.append(image_urls_to_embeddings(s=chunk, image_folder=image_folder, processor=processor, model=model))
    return np.concatenate(chunks)
