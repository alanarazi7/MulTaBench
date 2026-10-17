"""DINOv3 LoRA fine-tuning on (image, label), with finetune/lora.py."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from pandas import Series
from transformers import AutoImageProcessor, AutoModel, DINOv3ViTModel

from multabench.dino.constants import DINO_LORA, DINOV3_SMALL, LORA_IMAGE_TARGET_MODULES
from multabench.dino.image_loading import load_images
from multabench.finetune.lora import TunableEncoder, finetune_with_lora, finetuning_split
from multabench.finetune.train_args import LoraArgs
from multabench.utils.warnings import suppress_channel_dimension_warning


def _load_fresh_dino(device: Union[torch.device, str], model_name: str = DINOV3_SMALL) -> DINOv3ViTModel:
    """A new DINO instance, so LoRA can adapt it without affecting other uses."""
    return AutoModel.from_pretrained(model_name).to(device)


def cls_token(last_hidden: torch.Tensor) -> torch.Tensor:
    return last_hidden[:, 0, :]


def finetune_dino(files: Series, y: Series, is_cls: bool, image_folder: str, device: Union[torch.device, str],
                  model_name: str = DINOV3_SMALL, args: LoraArgs = DINO_LORA) -> Tuple[nn.Module, Any, dict]:
    """DINO tuned on the training rows' images, each labeled with its row's target."""
    x_tr, x_val, y_tr, y_val = finetuning_split(x=files.to_frame(), y=y, is_cls=is_cls)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    def featurize(image: Image.Image) -> Dict[str, torch.Tensor]:
        with suppress_channel_dimension_warning():
            return {"pixel_values": processor(images=[image], return_tensors="pt")["pixel_values"].squeeze(0)}

    encoder = TunableEncoder(
        load_backbone=lambda: _load_fresh_dino(device, model_name=model_name),
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
