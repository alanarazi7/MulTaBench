from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from pandas import DataFrame, Series

from multabench.baselines.preprocessing.column_encoders import ColumnEncoder, column_caches, fit_column_encoders
from multabench.benchmark.splits import Size
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.dino.constants import IMAGE_ENCODERS, ImageEncoder
from multabench.dino.dino_finetune import encode_image_files, finetune_dino, get_vanilla_dino
from multabench.embeddings.hub import CACHED_ENCODERS, CachedEmbeddings


class DinoColumnEncoder(ColumnEncoder):
    SUFFIX = "img"
    SCALE = True

    def __init__(self, col: str, model: Any, processor: Any, image_folder: str,
                 cache: Optional[CachedEmbeddings] = None):
        super().__init__(col, cache=cache)
        self.model = model
        self.processor = processor
        self.image_folder = image_folder

    def encode(self, values: Series) -> np.ndarray:
        return encode_image_files(values, image_folder=self.image_folder, model=self.model, processor=self.processor)

    def cache_keys(self, values: Series) -> List[str]:
        return values.astype(str).tolist()

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return super().transform(embeddings).astype(np.float32)


def fit_image_encoders(x: DataFrame, image_features: Set[str], device: torch.device, image_folder: Optional[str],
                       encoder: ImageEncoder = ImageEncoder.DINO_SMALL, y: Optional[Series] = None, is_cls: bool = True,
                       dataset: Optional[MulTaBenchDatasetID] = None, size: Optional[Size] = None,
                       fold: Optional[int] = None) -> Tuple[Dict[str, ColumnEncoder], Dict[str, np.ndarray]]:
    """One encoder per image column, with the training rows' embeddings. A tuned encoder is DINO fine-tuned on the
    first image column of the training rows (y) unless the fold's tuned embeddings are cached."""
    columns = [c for c in x.columns if c in image_features]
    if not columns:
        return {}, {}
    if image_folder is None:
        raise ValueError(f"Image columns {columns} need an image_folder")
    spec = IMAGE_ENCODERS[encoder]
    caches = column_caches(encoder, dataset, columns, size=size, fold=fold) if encoder in CACHED_ENCODERS else {}
    model, processor = None, None
    if not set(caches).issuperset(columns):
        if spec.tune_encoder:
            model, processor, _ = finetune_dino(x[columns[0]], y, is_cls=is_cls, image_folder=image_folder,
                                                device=device, model_name=spec.encoder_name)
        else:
            model, processor = get_vanilla_dino(spec.encoder_name)
            model.to(device)
    return fit_column_encoders(x, [DinoColumnEncoder(col, model, processor, image_folder, cache=caches.get(col))
                                   for col in columns])
