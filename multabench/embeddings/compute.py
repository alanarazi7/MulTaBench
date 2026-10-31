"""Compute the cached embeddings exactly as a benchmark run embeds its features live, and write them to disk."""
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series

from multabench.baselines.preprocessing.feature_types import detect_feature_types, transform_feature_types
from multabench.benchmark.load import split_dataset
from multabench.benchmark.splits import Size
from multabench.datasets.objects import MultimodalDataset
from multabench.encoders.constants import ImageEncoder, TextEncoder
from multabench.encoders.dino.constants import DINO_LORA
from multabench.encoders.dino.dino_encoder import get_image_encoder, image_urls_to_embeddings
from multabench.encoders.dino.dino_finetune import finetune_dino
from multabench.encoders.e5.constants import E5_LORA
from multabench.encoders.e5.e5_finetune import encode_texts_with_e5, finetune_e5, get_vanilla_e5
from multabench.encoders.tile.tile_encoder import TileRows, get_tile_llm
from multabench.embeddings.hub import (CachedEncoder, encoder_inputs, encoder_spec, input_values, is_tuned,
                                       write_embeddings)
from multabench.utils.hardware import EMBEDDING_HARDWARE, get_hardware_dict, hardware_mismatches
from multabench.utils.logging import get_current_commit_hash
from multabench.utils.timing import pop_embedding_seconds

# Up to ~100K images are embedded, which don't fit in memory at once.
IMAGE_CHUNK_ROWS = 4096


def compute_embeddings(dataset: MultimodalDataset, encoder: CachedEncoder, path: str, device: torch.device,
                       size: Optional[Size] = None, fold: Optional[int] = None):
    """A frozen encoder embeds every row of the dataset. A tuned one is fine-tuned on the fold's training rows, as a run
    would, then embeds the fold's train and test rows; its LoRA adapter is saved next to the embeddings."""
    meta: Dict[str, Any] = {"dataset": dataset.dataset_id.name, "encoder": str(encoder),
                            "model": encoder_spec(encoder).encoder_name}
    if is_tuned(encoder):
        split = split_dataset(dataset, fold=fold, size=size)
        x = pd.concat([split.x_train, split.x_test])
        # Detected on the training rows, as a run detects them.
        features = _features(split.x_train, dataset, encoder)
        meta.update(size=str(Size(size)), fold=fold, n_train=len(split.x_train), n_test=len(split.x_test))
    else:
        x = dataset.x
        features = _features(x, dataset, encoder)
    hardware = {**get_hardware_dict(device), "expected_hardware": asdict(EMBEDDING_HARDWARE),
                "official_hardware_mismatches": hardware_mismatches(device, EMBEDDING_HARDWARE)}
    print(f"Embedding {len(features)} of {x.shape[1]} columns, {len(x)} rows, with {encoder} on {hardware}")
    pop_embedding_seconds()
    model, processor = None, None
    if features and is_tuned(encoder):
        model, processor, stats = _finetune(encoder, split.x_train, split.y_train, features, dataset, device)
        model.save_pretrained(path)
        meta.update(lora=(E5_LORA if isinstance(encoder, TextEncoder) else DINO_LORA).to_dict(), **stats)
    elif features:
        model, processor = _load_frozen(encoder, device)
    load_seconds = pop_embedding_seconds()
    if encoder == TextEncoder.TILE:
        inputs, embeddings, seconds = _encode_rows(x, dataset.image_column, features, device)
    else:
        inputs, embeddings, seconds = _encode(x, dataset.image_folder, encoder, features, model, processor, device)
    total_seconds = load_seconds + sum(seconds.values())
    meta.update(n_rows=len(x), n_inputs=len(inputs), n_columns=x.shape[1], features=features,
                **{"finetune_seconds" if is_tuned(encoder) else "load_seconds": load_seconds},
                encode_seconds=seconds, total_seconds=total_seconds, git=get_current_commit_hash(),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), **hardware)
    write_embeddings(path, inputs=inputs, embeddings=embeddings, meta=meta,
                     dtype=np.float16 if encoder == TextEncoder.TILE else np.float32)
    print(f"Embedded {dataset.dataset_id.name} with {encoder} in {total_seconds:.0f}s ({load_seconds:.0f}s "
          f"{'fine-tuning' if is_tuned(encoder) else 'loading'} the model)")


def _features(x: DataFrame, dataset: MultimodalDataset, encoder: CachedEncoder) -> List[str]:
    if isinstance(encoder, TextEncoder):
        return sorted(detect_feature_types(x, image_column=dataset.image_column).text_features)
    return [dataset.image_column] if dataset.image_column in x.columns else []


def _load_frozen(encoder: CachedEncoder, device: torch.device) -> Tuple[Any, Any]:
    model_name = encoder_spec(encoder).encoder_name
    if encoder == TextEncoder.TILE:
        return get_tile_llm(device)
    if isinstance(encoder, TextEncoder):
        return get_vanilla_e5(device, model_name=model_name)
    model, processor = get_image_encoder(model_name=model_name)
    return model.to(device), processor


def _finetune(encoder: CachedEncoder, x_train: DataFrame, y_train: Series, features: List[str],
              dataset: MultimodalDataset, device: torch.device) -> Tuple[Any, Any, dict]:
    """The encoder fine-tuned as a run fine-tunes it: on the training rows, texts after the missing-text fill."""
    model_name = encoder_spec(encoder).encoder_name
    if isinstance(encoder, TextEncoder):
        texts = transform_feature_types(x_train[features], numerical_features=set(), image_features=set())
        return finetune_e5(texts, y_train, features, is_cls=dataset.is_cls, device=device, model_name=model_name)
    return finetune_dino(x_train[features[0]], y_train, is_cls=dataset.is_cls, image_folder=dataset.image_folder,
                         device=device, model_name=model_name)


def _encode(x: DataFrame, image_folder: str, encoder: CachedEncoder, features: List[str], model: Any, processor: Any,
            device: torch.device) -> Tuple[List[str], np.ndarray, Dict[str, float]]:
    """Each feature's distinct values embedded once, keyed by the input the encoder got."""
    inputs, embeddings, seconds = [], [], {}
    for col in features:
        values = list(dict.fromkeys(input_values(x, encoder, col)))
        if isinstance(encoder, TextEncoder):
            embedded = encode_texts_with_e5(texts=values, col_name=str(col), model=model, tokenizer=processor,
                                            device=device)
        elif isinstance(encoder, ImageEncoder):
            embedded = _embed_images_in_chunks(Series(values), image_folder=image_folder, model=model,
                                               processor=processor)
        else:
            raise TypeError(f"Unknown encoder {encoder!r}")
        seconds[col] = pop_embedding_seconds()
        inputs += encoder_inputs(encoder, col, values)
        embeddings.append(embedded)
        print(f"Embedded {col}: {len(values)} distinct values of {len(x)} rows in {seconds[col]:.0f}s")
    return inputs, np.concatenate(embeddings), seconds


def _encode_rows(x: DataFrame, image_column: Optional[str], features: List[str],
                 device: torch.device) -> Tuple[List[str], np.ndarray, Dict[str, float]]:
    """TILE: each distinct row serialized and embedded once, keyed by its text, with all its text cells."""
    rows = TileRows(image_column=image_column, device=device)
    rows.add(x)
    rows.text_columns = features
    texts = Series([rows.texts[i] for i in x.index], index=x.index).drop_duplicates()
    cells = rows.cells(texts.index)
    seconds = {"rows": pop_embedding_seconds()}
    print(f"Embedded {len(texts)} distinct rows of {len(x)} in {seconds['rows']:.0f}s")
    return texts.tolist(), cells.transpose(1, 0, 2), seconds


def _embed_images_in_chunks(urls: Series, image_folder: str, model: Any, processor: Any) -> np.ndarray:
    chunks = []
    for start in range(0, len(urls), IMAGE_CHUNK_ROWS):
        chunk = urls.iloc[start:start + IMAGE_CHUNK_ROWS]
        chunks.append(image_urls_to_embeddings(s=chunk, image_folder=image_folder, processor=processor, model=model))
    return np.concatenate(chunks)
