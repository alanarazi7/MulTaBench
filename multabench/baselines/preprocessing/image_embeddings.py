from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from pandas import DataFrame, Series
from transformers import AutoImageProcessor, AutoModel, DINOv3ViTImageProcessorFast, DINOv3ViTModel

from multabench.baselines.preprocessing.column_encoders import ColumnEncoder, column_caches, fit_column_encoders
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.dino.constants import DINO_LORA, DINOV3_SMALL
from multabench.dino.dino_finetune import finetune_dino
from multabench.dino.image_loading import load_images
from multabench.embeddings.hub import CachedEmbeddings
from multabench.utils.timing import embedding_step
from multabench.utils.warnings import suppress_channel_dimension_warning

# Alternatives: facebook/dinov3-vits16plus-pretrain-lvd1689m, facebook/dinov3-convnext-tiny-pretrain-lvd1689m

IMAGE_BATCH_SIZE = 64

IMAGE_ENCODER = {"encoder": None, "processor": None}


class DinoColumnEncoder(ColumnEncoder):
    """DINO's CLS token, vanilla or tuned, standardized before PCA."""
    SUFFIX = "img"
    SCALE = True

    def __init__(self, col: str, model: Any, processor: Any, image_folder: str,
                 cache: Optional[CachedEmbeddings] = None):
        super().__init__(col, cache=cache)
        self.model = model
        self.processor = processor
        self.image_folder = image_folder

    def encode(self, values: Series) -> np.ndarray:
        return image_urls_to_embeddings(values, image_folder=self.image_folder, processor=self.processor,
                                        model=self.model)

    def cache_keys(self, values: Series) -> List[str]:
        return values.astype(str).tolist()

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return super().transform(embeddings).astype(np.float32)


def fit_image_encoders(x: DataFrame, image_features: Set[str], device: torch.device, image_folder: Optional[str],
                       y: Optional[Series] = None, is_cls: bool = True, tune_dino: bool = False,
                       dino_train_kwargs: Optional[Dict[str, Any]] = None, dino_model_name: str = DINOV3_SMALL,
                       dataset: Optional[MulTaBenchDatasetID] = None
                       ) -> Tuple[Dict[str, ColumnEncoder], Dict[str, np.ndarray]]:
    """One encoder per image column, with the training rows' embeddings. Tuning fine-tunes DINO on the first image
    column of the training rows (y); otherwise columns with cached embeddings on the Hub are read from there."""
    columns = [c for c in x.columns if c in image_features]
    if not columns:
        return {}, {}
    if image_folder is None:
        raise ValueError(f"Image columns {columns} need an image_folder")
    tune = tune_dino and y is not None
    caches = {} if tune else column_caches(dino_model_name, dataset)
    model, processor = None, None
    if tune:
        model, processor, _ = finetune_dino(x[columns[0]], y, is_cls=is_cls, image_folder=image_folder, device=device,
                                            model_name=dino_model_name,
                                            args=replace(DINO_LORA, **(dino_train_kwargs or {})))
    elif not set(caches).issuperset(columns):
        model, processor = get_image_encoder(model_name=dino_model_name)
        model.to(device)
    return fit_column_encoders(x, [DinoColumnEncoder(col, model, processor, image_folder, cache=caches.get(col))
                                   for col in columns])


@embedding_step
def get_image_encoder(model_name: str) -> Tuple[DINOv3ViTModel, DINOv3ViTImageProcessorFast]:
    if IMAGE_ENCODER["encoder"] is None:
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        IMAGE_ENCODER["encoder"] = model
        IMAGE_ENCODER["processor"] = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    dino_model = IMAGE_ENCODER["encoder"]
    img_processor = IMAGE_ENCODER["processor"]
    if not isinstance(dino_model, DINOv3ViTModel):
        raise TypeError(f"Expected DINOv3ViTModel, got {type(dino_model)}")
    if not isinstance(img_processor, (DINOv3ViTImageProcessorFast)):
        raise TypeError(f"Expected BitImageProcessor, got {type(img_processor)}")
    return dino_model, img_processor


def image_urls_to_embeddings(
    s: Series,
    image_folder: str,
    processor: DINOv3ViTImageProcessorFast,
    model: DINOv3ViTModel,
) -> np.ndarray:
    images = load_images(s=s, image_folder=image_folder)
    embeddings = encode_images_in_batches(images=images, processor=processor, model=model)
    return embeddings


@embedding_step
def encode_images_in_batches(
    images: List[Image.Image],
    processor: DINOv3ViTImageProcessorFast,
    model: DINOv3ViTModel,
) -> np.ndarray:
    embeddings = []
    for i in range(0, len(images), IMAGE_BATCH_SIZE):
        batch_images = images[i:i + IMAGE_BATCH_SIZE]
        with suppress_channel_dimension_warning():
            inputs = processor(images=batch_images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden = outputs["last_hidden_state"] if isinstance(outputs, dict) else outputs.last_hidden_state
        batch_embeddings = last_hidden[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    embeddings = np.concatenate(embeddings, axis=0)
    d_model = model.config.hidden_size
    assert embeddings.shape == (len(images), d_model), f"Expected ({len(images)}, {d_model}), got {embeddings.shape}"
    return embeddings
