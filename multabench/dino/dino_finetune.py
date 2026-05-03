"""Finetune DINOv3 with LoRA on (images, label). Uses Hugging Face Trainer + PEFT."""

from __future__ import annotations

import os
import re
import tempfile
from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModel,
    DINOv3ViTImageProcessorFast,
    DINOv3ViTModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    EvalPrediction,
)

from tabstar.constants import SEED
from multabench.finetune.utils import compute_metrics_multiclass, encoder_finetune_loss
from multabench.dino.constants import DINO_DIM, DINO_NUM_LAYERS, DINOV3_SMALL, LORA_IMAGE_TARGET_MODULES
from multabench.dino.image_loading import load_images
from multabench.utils.warnings import suppress_channel_dimension_warning


def _get_lora_target_modules(layer_indices: list[int]) -> list[str]:
    target_modules = []
    for layer_idx in layer_indices:
        for module_name in LORA_IMAGE_TARGET_MODULES:
            target_modules.append(f"layer.{layer_idx}.attention.{module_name}")
    return target_modules


def _print_trainable_params_per_layer(model: nn.Module) -> None:
    layer_re = re.compile(r"layer\.(\d+)")
    groups: dict[str, int] = {}
    total_trainable, total_frozen = 0, 0
    for name, p in model.named_parameters():
        n = p.numel()
        if not p.requires_grad:
            total_frozen += n
            continue
        total_trainable += n
        m = layer_re.search(name)
        key = f"layer.{int(m.group(1))}" if m else ("head" if "head" in name else "other")
        groups[key] = groups.get(key, 0) + n
    for key in sorted(groups, key=lambda k: (0, int(k.split(".")[1])) if k.startswith("layer.") else (1, 0)):
        print(f"  {key}: {groups[key]:,}")
    print(f"  total trainable: {total_trainable:,}  (frozen: {total_frozen:,})")


class ImageLabelDataset(Dataset):
    """Dataset of (images, labels) for Trainer. Labels are always class indices (long): from label_encoder for classification, or bin indices for regression."""

    def __init__(
        self,
        image_paths: List[Union[str, Image.Image]],
        labels: np.ndarray,
        processor: DINOv3ViTImageProcessorFast,
        label_encoder: Optional[LabelEncoder] = None,
    ):
        self.image_paths = image_paths
        self.processor = processor
        if label_encoder is not None:
            enc = label_encoder.transform(labels.ravel().astype(str))
        else:
            enc = labels.ravel().astype(int)
        self.labels = torch.tensor(enc, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with suppress_channel_dimension_warning():
            item = self.image_paths[idx]
            if isinstance(item, Image.Image):
                images = [item]
            else:
                # item is a path (str); load_images expects iterable of paths
                images = load_images([item])
            inputs = self.processor(images=images, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": self.labels[idx],
        }


class DINOForTuning(nn.Module):
    """DINO backbone + head for classification. Same architecture always: d_output = number of unique labels (classes or bins). CE loss."""

    def __init__(
        self,
        backbone: DINOv3ViTModel,
        d_output: int,
        d_model: int = DINO_DIM[DINOV3_SMALL],
    ):
        super().__init__()
        self.d_output = d_output
        self.backbone = backbone
        self.head = nn.Linear(d_model, d_output)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values, **kwargs)
        cls_token = outputs.last_hidden_state[:, 0, :]
        if cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)
        logits = self.head(cls_token)
        if labels is not None:
            loss = encoder_finetune_loss(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def _load_fresh_dino(device: torch.device, model_name: str = DINOV3_SMALL) -> Tuple[DINOv3ViTModel, DINOv3ViTImageProcessorFast]:
    """Load a fresh DINO model and processor. Each call returns a new independent instance, so LoRA weights can be applied without affecting other uses."""
    model = AutoModel.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model.to(device)
    return model, processor


def finetune_dino_with_lora(
    train_images: List[Image.Image],
    train_y: np.ndarray,
    val_images: List[Image.Image],
    val_y: np.ndarray,
    device: Union[torch.device, str],
    processor: DINOv3ViTImageProcessorFast,
    *,
    model_name: str = DINOV3_SMALL,
    lora_rank: int = 16,
    img_layers: int = 3,
    learning_rate: float = 1e-4,
    epochs: int = 40,
    patience: int = 3,
    batch_size: int = 512,
    weight_decay: float = 0.01,
    dataloader_num_workers: int = 4,
    seed: int = SEED,
    save_path: str | None = None,
) -> Tuple[DINOv3ViTModel, DINOv3ViTImageProcessorFast]:
    """
    Finetune DINOv3 with LoRA using Hugging Face Trainer. Uses train/val for early stopping.
    Single GPU only (device e.g. torch.device("cuda", 3) or "cuda:3"). No model/data parallelism.
    Returns the backbone (encoding outputs D_DINO dims, PCA applied downstream).
    """
    assert "CUDA_VISIBLE_DEVICES" in os.environ, "CUDA_VISIBLE_DEVICES must be set"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Input shapes and dataset sizes (single device, no parallelism) ---
    n_train, n_val = len(train_images), len(val_images)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    print(f"[Dataset] train: n={n_train}, val: n={n_val}")
    print(f"[Dataset] train_y.shape={train_y.shape}, val_y.shape={val_y.shape} (y = labels)")
    print(f"[Steps] batch_size={batch_size} → {steps_per_epoch} steps/epoch → {total_steps} total steps ({epochs} epochs)")

    dino_model, _ = _load_fresh_dino(device, model_name=model_name)
    d_model, num_layers = DINO_DIM[model_name], DINO_NUM_LAYERS[model_name]

    label_encoder = LabelEncoder()
    label_encoder.fit(train_y.ravel().astype(str))
    d_output_inferred = len(label_encoder.classes_)
    train_ds = ImageLabelDataset(train_images, train_y, processor, label_encoder)
    val_ds = ImageLabelDataset(val_images, val_y, processor, label_encoder)
    # LoRA on last N layers
    n_layers = min(max(1, img_layers), num_layers)
    print(f"🦜 Doing LoRA on last {n_layers} layers of {model_name} with rank {lora_rank}")
    layers_to_adapt = list(range(num_layers))[-n_layers:]
    target_modules = _get_lora_target_modules(layers_to_adapt)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    dino_model = get_peft_model(dino_model, lora_config)
    model = DINOForTuning(backbone=dino_model, d_output=d_output_inferred, d_model=d_model)
    _print_trainable_params_per_layer(model)

    with tempfile.TemporaryDirectory(prefix="dino_finetune_") as tmpdir:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            seed=seed,
            remove_unused_columns=False,
            report_to="none",
            dataloader_num_workers=dataloader_num_workers,
        )

        def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
            return compute_metrics_multiclass(eval_pred, d_output=d_output_inferred)

        trainer_kw: Dict[str, Any] = dict(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )
        # Trainer will use the only visible GPU (cuda:0); no DataParallel
        trainer = Trainer(**trainer_kw)
        trainer.train()
        # Trainer loads best model at end when load_best_model_at_end=True
        backbone = trainer.model.backbone
        if save_path is not None:
            backbone.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"💾 Saved fine-tuned DINO to {save_path}")

    backbone.eval()
    return backbone, processor
