import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List, Any, Set

import pandas as pd
import torch
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from multabench.constants import SEED
from transformers import AutoImageProcessor, AutoModel, DINOv3ViTImageProcessorFast, DINOv3ViTModel

from multabench.dino.constants import DINOV3_SMALL
from multabench.dino.dino_finetune import finetune_dino_with_lora
from multabench.dino.image_loading import load_images
from multabench.preprocessing.discretize import discretize_numerical
from multabench.utils.warnings import suppress_channel_dimension_warning
from multabench.preprocessing.splits import split_to_val
from multabench.utils.timing import embedding_step

# Alternatives: facebook/dinov3-vits16plus-pretrain-lvd1689m, facebook/dinov3-convnext-tiny-pretrain-lvd1689m

PCA_COMPONENTS = 30
IMAGE_BATCH_SIZE = 64

IMAGE_ENCODER = {"encoder": None, "processor": None}


class _ScaledPCA:
    """StandardScaler then PCA; exposes .transform(X) and .n_components for drop-in use."""
    def __init__(self, scaler: StandardScaler, pca: PCA):
        self.scaler = scaler
        self.pca = pca
        self.n_components = pca.n_components

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
) -> Tuple[Dict[str, Any], Optional[Any], Optional[DINOv3ViTImageProcessorFast], Dict[str, np.ndarray]]:
    """
    Fit PCA per image column. If tune_dino and y is provided,
    finetune DINO with LoRA on train/val split then use the best model for encoding.
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
        tuned_model, tuned_processor = finetune_dino_on_column(x=x, image_col=image_features[0], y=y, device=device,
                                                               image_folder=image_folder, is_cls=is_cls,
                                                               dino_train_kwargs=dino_train_kwargs,
                                                               dino_model_name=dino_model_name)
        dino_model, img_processor = tuned_model, tuned_processor
    else:
        dino_model, img_processor = get_image_encoder(model_name=dino_model_name)
        dino_model.to(device)

    for col in image_features:
        images = load_images(s=x[col], image_folder=image_folder)
        embeddings = encode_images_in_batches(images=images, processor=img_processor, model=dino_model)
        scaler = StandardScaler()
        scaler.fit(embeddings)
        scaled = scaler.transform(embeddings)
        pca_col = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        pca_col.fit(scaled)
        image_encoders[col] = _ScaledPCA(scaler, pca_col)
        train_embeddings[col] = embeddings

    return image_encoders, tuned_model, tuned_processor, train_embeddings

def finetune_dino_on_column(
    x: DataFrame,
    image_col: str,
    y: Series,
    device: torch.device,
    image_folder: str,
    is_cls: bool,
    dino_train_kwargs: Optional[Dict[str, Any]] = None,
    dino_model_name: str = DINOV3_SMALL,
) -> Tuple[DINOv3ViTModel, DINOv3ViTImageProcessorFast]:
    if not is_cls:
        y = discretize_numerical(y, n_bins=20)
    x_tr, x_val, y_tr, y_val = split_to_val(x=x, y=y, is_cls=True)
    train_imgs = load_images(s=x_tr[image_col], image_folder=image_folder)
    val_imgs = load_images(s=x_val[image_col], image_folder=image_folder)
    kwargs = dino_train_kwargs or {}
    tuned_model, tuned_processor = finetune_dino_with_lora(
        train_images=train_imgs,
        train_y=y_tr.values,
        val_images=val_imgs,
        val_y=y_val.values,
        device=device,
        processor=AutoImageProcessor.from_pretrained(dino_model_name, use_fast=True),
        model_name=dino_model_name,
        **kwargs,
    )
    tuned_model.to(device)
    return tuned_model, tuned_processor


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
    if dino_model is not None and dino_processor is not None:
        pass  # use provided tuned model
    else:
        dino_model, dino_processor = get_image_encoder(model_name=dino_model_name)
    dino_model.to(device)
    img_processor = dino_processor
    train_embeddings = train_embeddings or {}
    for image_col, image_pca in image_transformers.items():
        assert image_col in x.columns, f"Image column {image_col} not found in DataFrame"
        s = x[image_col]
        embeddings = train_embeddings.get(image_col)
        if embeddings is None:
            embeddings = image_urls_to_embeddings(s=x[image_col], image_folder=image_folder, processor=img_processor, model=dino_model)
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
