from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pandas import DataFrame, Series
from transformers import AutoImageProcessor, AutoModel, DINOv3ViTImageProcessorFast, DINOv3ViTModel

from multabench.embeddings.hub import CachedEmbeddings
from multabench.encoders.abstract_encoder import EmbeddingEncoder
from multabench.encoders.column_encoders import ColumnEncoder
from multabench.encoders.dino.dino_finetune import finetune_dino
from multabench.encoders.dino.image_loading import load_images
from multabench.utils.timing import embedding_step
from multabench.utils.warnings import suppress_channel_dimension_warning

# Alternatives: facebook/dinov3-vits16plus-pretrain-lvd1689m, facebook/dinov3-convnext-tiny-pretrain-lvd1689m

IMAGE_BATCH_SIZE = 64

IMAGE_ENCODER = {"encoder": None, "processor": None}


class DinoColumnEncoder(ColumnEncoder):
    """DINO's CLS token, vanilla or tuned; standardized, then PCA."""
    SUFFIX = "img"

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


class DinoEncoder(EmbeddingEncoder):
    """DINO on each image column, vanilla or, for a tuned encoder, fine-tuned on (x, y) of the first one."""

    def column_encoders(self, x: DataFrame, columns: List[str], y: Optional[Series],
                        is_cls: bool) -> List[ColumnEncoder]:
        if self.image_folder is None:
            raise ValueError(f"Image columns {columns} need an image_folder")
        caches = self.caches(columns)
        model, processor = None, None
        if not set(caches).issuperset(columns):
            if self.spec.tune_encoder:
                model, processor, _ = finetune_dino(x[columns[0]], y, is_cls=is_cls, image_folder=self.image_folder,
                                                    device=self.device, model_name=self.spec.encoder_name)
            else:
                model, processor = get_image_encoder(model_name=self.spec.encoder_name)
                model.to(self.device)
        return [DinoColumnEncoder(col, model, processor, self.image_folder, cache=caches.get(col)) for col in columns]


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
