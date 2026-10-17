from dataclasses import replace

import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List, Any, Set

import pandas as pd
import torch
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from multabench.constants import SEED
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from transformers import AutoImageProcessor, AutoModel, DINOv3ViTImageProcessorFast, DINOv3ViTModel

from multabench.dino.constants import DINO_LORA, DINOV3_SMALL
from multabench.dino.dino_finetune import finetune_dino
from multabench.dino.image_loading import load_images
from multabench.embeddings.hub import CachedEmbeddings, cached_embeddings, cached_encoder
from multabench.utils.warnings import suppress_channel_dimension_warning
from multabench.utils.timing import embedding_step

# Alternatives: facebook/dinov3-vits16plus-pretrain-lvd1689m, facebook/dinov3-convnext-tiny-pretrain-lvd1689m

PCA_COMPONENTS = 30
IMAGE_BATCH_SIZE = 64

IMAGE_ENCODER = {"encoder": None, "processor": None}


class _ScaledPCA:
    """StandardScaler then PCA; exposes .transform(X) and .n_components for drop-in use. Holds the column's cached
    embeddings when it has them."""
    def __init__(self, scaler: StandardScaler, pca: PCA, cache: Optional[CachedEmbeddings] = None):
        self.scaler = scaler
        self.pca = pca
        self.n_components = pca.n_components
        self.cache = cache

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))


def fit_image_encoders(
    x: DataFrame,
    image_features: Set[str],
    device: torch.device,
    image_folder: Optional[str],
    is_cls: bool,
    d_output: int,
    y: Optional[Series] = None,
    tune_dino: bool = False,
    dino_train_kwargs: Optional[Dict[str, Any]] = None,
    dino_model_name: str = DINOV3_SMALL,
    dataset: Optional[MulTaBenchDatasetID] = None,
) -> Tuple[Dict[str, Any], Optional[Any], Optional[DINOv3ViTImageProcessorFast], Dict[str, np.ndarray]]:
    """
    Fit PCA per image column. If tune_dino and y is provided,
    finetune DINO with LoRA on train/val split then use the best model for encoding.
    Otherwise columns with cached embeddings on the Hub are read from there instead of being encoded.
    Returns (image_encoders, tuned_dino_model_or_none, tuned_processor_or_none, train_embeddings per column).
    """
    image_encoders: Dict[str, Any] = {}
    train_embeddings: Dict[str, np.ndarray] = {}
    image_features = [c for c in x.columns if c in image_features]
    tuned_model = None
    tuned_processor: Optional[DINOv3ViTImageProcessorFast] = None

    if not image_features:
        return image_encoders, tuned_model, tuned_processor, train_embeddings
    if image_folder is None:
        raise ValueError(f"Image columns {image_features} need an image_folder")

    if tune_dino and y is not None:
        tuned_model, tuned_processor, _ = finetune_dino(x[image_features[0]], y, is_cls=is_cls,
                                                        image_folder=image_folder, device=device,
                                                        model_name=dino_model_name,
                                                        args=replace(DINO_LORA, **(dino_train_kwargs or {})))
        dino_model, img_processor = tuned_model, tuned_processor
    caches = {} if tune_dino else _caches_per_column(dino_model_name, dataset)
    if not tune_dino and not set(caches).issuperset(image_features):
        dino_model, img_processor = get_image_encoder(model_name=dino_model_name)
        dino_model.to(device)

    for col in image_features:
        cache = caches.get(col)
        if cache is None:
            images = load_images(s=x[col], image_folder=image_folder)
            embeddings = encode_images_in_batches(images=images, processor=img_processor, model=dino_model)
        else:
            embeddings = cache.lookup(x[col].astype(str).tolist(), feature=col)
        scaler = StandardScaler()
        scaler.fit(embeddings)
        scaled = scaler.transform(embeddings)
        pca_col = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        pca_col.fit(scaled)
        image_encoders[col] = _ScaledPCA(scaler, pca_col, cache=cache)
        train_embeddings[col] = embeddings

    return image_encoders, tuned_model, tuned_processor, train_embeddings


def _caches_per_column(dino_model_name: str, dataset: Optional[MulTaBenchDatasetID]) -> Dict[str, CachedEmbeddings]:
    """The cached embeddings of each image column that has them."""
    encoder = cached_encoder(dino_model_name, tuned=False)
    if encoder is None or dataset is None:
        return {}
    cache = cached_embeddings(encoder, dataset)
    print(f"Reading the image embeddings of {cache.meta['features']} from the cached {encoder} embeddings")
    return {col: cache for col in cache.meta["features"]}


def transform_image_features(
    x: DataFrame,
    image_transformers: Dict[str, Any],
    device: torch.device,
    image_folder: str,
    dino_model: Optional[DINOv3ViTModel] = None,
    dino_processor: Optional[DINOv3ViTImageProcessorFast] = None,
    dino_model_name: str = DINOV3_SMALL,
    train_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> DataFrame:
    # TODO: In realistic scenarios where we tune the model repeatedly, the efficient way would be to fit the encoder once,
    # cache the embeddings, and then apply PCA on the cached embeddings over and over for every train split in every run.
    if not image_transformers:
        return x
    train_embeddings = train_embeddings or {}
    for image_col, image_pca in image_transformers.items():
        assert image_col in x.columns, f"Image column {image_col} not found in DataFrame"
        s = x[image_col]
        embeddings = train_embeddings.get(image_col)
        if embeddings is None and image_pca.cache is not None:
            embeddings = image_pca.cache.lookup(s.astype(str).tolist(), feature=image_col)
        if embeddings is None:
            if dino_model is None or dino_processor is None:
                dino_model, dino_processor = get_image_encoder(model_name=dino_model_name)
            dino_model.to(device)
            embeddings = image_urls_to_embeddings(s=s, image_folder=image_folder, processor=dino_processor, model=dino_model)
        assert len(embeddings) == len(x)
        pca_vec = image_pca.transform(embeddings).astype(np.float32)
        pca_cols = [f"{image_col}_img_pca_{i}" for i in range(pca_vec.shape[1])]
        pca_df = pd.DataFrame(pca_vec, index=s.index, columns=pca_cols)
        x = x.drop(columns=[image_col])
        x = pd.concat([x, pca_df], axis=1)
    return x


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
