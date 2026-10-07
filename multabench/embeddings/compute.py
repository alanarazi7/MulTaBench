"""Compute the cached embeddings exactly as a benchmark run embeds its features live, and write them to disk."""
import time
from os.path import join
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pandas import DataFrame

from multabench.baselines.preprocessing.feature_types import detect_feature_types, transform_feature_types
from multabench.baselines.preprocessing.image_embeddings import (finetune_dino_on_column, get_image_encoder,
                                                                 image_urls_to_embeddings)
from multabench.baselines.preprocessing.text_embeddings import finetune_e5_on_columns
from multabench.benchmark.splits import get_split
from multabench.datasets.objects import MultimodalDataset
from multabench.e5.constants import TextEncoder
from multabench.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5
from multabench.embeddings.hub import ADAPTER_DIR, CachedEncoder, encoder_spec, is_tuned, write_embeddings
from multabench.finetune.train_args import DinoTrainArgs, E5TrainArgs
from multabench.utils.hardware import get_hardware_dict
from multabench.utils.logging import get_current_commit_hash
from multabench.utils.timing import pop_embedding_seconds

# Frozen encoders embed every row, up to ~100K images, which don't fit in memory at once.
IMAGE_CHUNK_ROWS = 4096


def compute_embeddings(dataset: MultimodalDataset, encoder: CachedEncoder, path: str, device: torch.device,
                       size: Optional[str] = None, fold: Optional[int] = None):
    features = _features(dataset, encoder)
    meta: Dict[str, Any] = {"dataset": dataset.dataset_id.name, "encoder": str(encoder),
                            "model": encoder_spec(encoder).encoder_name}
    pop_embedding_seconds()
    if is_tuned(encoder):
        train_idx, test_idx = get_split(dataset.y.to_numpy(), is_cls=dataset.is_cls, split=fold, size=size)
        rows = np.sort(np.concatenate([train_idx, test_idx]))
        train_kwargs = _train_kwargs(encoder)
        model, processor = _finetune(dataset, encoder, features, train_idx, device, train_kwargs) if features else (None, None)
        meta.update(size=size, fold=fold, n_train=len(train_idx), n_test=len(test_idx), train_kwargs=train_kwargs,
                    finetune_seconds=pop_embedding_seconds())
    else:
        rows = np.arange(len(dataset.y))
        model, processor = _load_frozen(encoder, device) if features else (None, None)
        meta.update(load_seconds=pop_embedding_seconds())
    embeddings, seconds = _encode(dataset, encoder, features, rows, model, processor, device)
    meta.update(n_rows=len(rows), encode_seconds=seconds, git=get_current_commit_hash(),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), **get_hardware_dict(device))
    if is_tuned(encoder) and model is not None:
        model.save_pretrained(join(path, ADAPTER_DIR))
    write_embeddings(path, rows=rows, embeddings=embeddings, meta=meta)


def _features(dataset: MultimodalDataset, encoder: CachedEncoder) -> List[str]:
    if isinstance(encoder, TextEncoder):
        return sorted(detect_feature_types(dataset.x, image_column=dataset.image_column).text_features)
    return [dataset.image_column] if dataset.image_column in dataset.x.columns else []


def _text_frame(x: DataFrame, features: List[str]) -> DataFrame:
    # A run fills missing text before embedding it, so the cache embeds the same strings.
    return transform_feature_types(x[features], numerical_features=set(), image_features=set())


def _train_kwargs(encoder: CachedEncoder) -> Dict[str, Any]:
    return (E5TrainArgs() if isinstance(encoder, TextEncoder) else DinoTrainArgs()).to_dict()


def _load_frozen(encoder: CachedEncoder, device: torch.device) -> Tuple[Any, Any]:
    model_name = encoder_spec(encoder).encoder_name
    if isinstance(encoder, TextEncoder):
        return get_vanilla_e5(device, model_name=model_name)
    model, processor = get_image_encoder(model_name=model_name)
    return model.to(device), processor


def _finetune(dataset: MultimodalDataset, encoder: CachedEncoder, features: List[str], train_idx: np.ndarray,
              device: torch.device, train_kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
    model_name = encoder_spec(encoder).encoder_name
    x_train, y_train = dataset.x.iloc[train_idx], dataset.y.iloc[train_idx]
    if isinstance(encoder, TextEncoder):
        return finetune_e5_on_columns(x=_text_frame(x_train, features), text_features_list=features, device=device,
                                      y=y_train, e5_train_kwargs=train_kwargs, is_cls=dataset.is_cls,
                                      e5_model_name=model_name)
    return finetune_dino_on_column(x=x_train, image_col=features[0], y=y_train, device=device,
                                   image_folder=dataset.image_folder, is_cls=dataset.is_cls,
                                   dino_train_kwargs=train_kwargs, dino_model_name=model_name)


def _encode(dataset: MultimodalDataset, encoder: CachedEncoder, features: List[str], rows: np.ndarray, model: Any,
            processor: Any, device: torch.device) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    embeddings, seconds = {}, {}
    x = dataset.x.iloc[rows]
    if isinstance(encoder, TextEncoder):
        x = _text_frame(x, features)
    for col in features:
        if isinstance(encoder, TextEncoder):
            embeddings[col] = encode_texts_with_e5(texts=x[col].tolist(), col_name=str(col), model=model,
                                                   tokenizer=processor, device=device)
        else:
            chunks = [image_urls_to_embeddings(s=x[col].iloc[i:i + IMAGE_CHUNK_ROWS], image_folder=dataset.image_folder,
                                               processor=processor, model=model)
                      for i in range(0, len(x), IMAGE_CHUNK_ROWS)]
            embeddings[col] = np.concatenate(chunks)
        seconds[col] = pop_embedding_seconds()
        print(f"Embedded {col}: {embeddings[col].shape} in {seconds[col]:.0f}s")
    return embeddings, seconds
