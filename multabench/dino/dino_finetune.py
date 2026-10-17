"""DINOv3 image encoding, frozen or fine-tuned with LoRA on the task's labels."""
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pandas import Series
from transformers import AutoImageProcessor, AutoModel, DINOv3ViTImageProcessorFast

from multabench.dino.constants import DINOV3_SMALL, LORA_IMAGE_TARGET_MODULES
from multabench.dino.image_loading import load_images
from multabench.finetune.lora import DINO_LORA, LoraArgs, TunableEncoder, finetune_with_lora, finetuning_split
from multabench.utils.timing import embedding_step
from multabench.utils.warnings import suppress_channel_dimension_warning

ENCODE_BATCH_SIZE = 64
# Up to ~100K images are embedded, which don't fit in memory at once.
LOAD_CHUNK_ROWS = 4096

Dino = Tuple[nn.Module, DINOv3ViTImageProcessorFast]

_VANILLA_DINO: Dict[str, Dino] = {}


def load_dino(model_name: str = DINOV3_SMALL) -> Dino:
    """A fresh model, so LoRA can adapt it without touching the shared frozen one."""
    return AutoModel.from_pretrained(model_name), AutoImageProcessor.from_pretrained(model_name, use_fast=True)


@embedding_step
def get_vanilla_dino(model_name: str = DINOV3_SMALL) -> Dino:
    if model_name not in _VANILLA_DINO:
        model, processor = load_dino(model_name)
        _VANILLA_DINO[model_name] = model.eval(), processor
    return _VANILLA_DINO[model_name]


def cls_token(last_hidden: torch.Tensor) -> torch.Tensor:
    return last_hidden[:, 0, :]


@embedding_step
def encode_images(images: List[Image.Image], model: nn.Module, processor: DINOv3ViTImageProcessorFast) -> np.ndarray:
    """The CLS token of each image; (N, d_model)."""
    embeddings = []
    for i in range(0, len(images), ENCODE_BATCH_SIZE):
        with suppress_channel_dimension_warning():
            inputs = processor(images=images[i:i + ENCODE_BATCH_SIZE], return_tensors="pt").to(model.device)
        with torch.no_grad():
            embeddings.append(cls_token(model(**inputs).last_hidden_state).cpu().numpy())
    embeddings = np.concatenate(embeddings)
    assert embeddings.shape == (len(images), model.config.hidden_size), embeddings.shape
    return embeddings


def encode_image_files(files: Series, image_folder: str, model: nn.Module,
                       processor: DINOv3ViTImageProcessorFast) -> np.ndarray:
    """Loads LOAD_CHUNK_ROWS images at a time, to bound memory."""
    return np.concatenate([encode_images(load_images(files.iloc[i:i + LOAD_CHUNK_ROWS], image_folder=image_folder),
                                         model=model, processor=processor)
                           for i in range(0, len(files), LOAD_CHUNK_ROWS)])


def finetune_dino(files: Series, y: Series, is_cls: bool, image_folder: str, device: Union[torch.device, str],
                  model_name: str = DINOV3_SMALL, args: LoraArgs = DINO_LORA) -> Tuple[nn.Module, Any, dict]:
    """DINO tuned on the training rows' images, each labeled with its row's target."""
    x_tr, x_val, y_tr, y_val = finetuning_split(x=files.to_frame(), y=y, is_cls=is_cls)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    def featurize(image: Image.Image) -> Dict[str, torch.Tensor]:
        with suppress_channel_dimension_warning():
            return {"pixel_values": processor(images=[image], return_tensors="pt")["pixel_values"].squeeze(0)}

    encoder = TunableEncoder(
        load_backbone=lambda: load_dino(model_name)[0].to(device),
        layer_modules=lambda layer: [f"layer.{layer}.{m}" for m in LORA_IMAGE_TARGET_MODULES],
        featurize=featurize,
        pool=lambda last_hidden, inputs: cls_token(last_hidden))
    col = files.name
    print(f"Fine-tuning {model_name} for the image column {col!r}")
    train_images = load_images(x_tr[col], image_folder=image_folder)
    val_images = load_images(x_val[col], image_folder=image_folder)
    backbone, stats = finetune_with_lora(encoder, train_images, y_tr.to_numpy(), val_images, y_val.to_numpy(),
                                         args=args)
    return backbone.to(device), processor, stats
