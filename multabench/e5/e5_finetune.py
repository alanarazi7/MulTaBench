"""Finetune E5-small-v2 with LoRA on (text, label). Uses Hugging Face Trainer + PEFT."""

from __future__ import annotations

import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from tabstar.constants import SEED
from multabench.finetune.utils import compute_metrics_multiclass, encoder_finetune_loss
from multabench.e5.constants import (
    E5_DIM,
    E5_NUM_LAYERS,
    E5_PASSAGE_PREFIX,
    E5_SMALL_V2,
    LORA_TEXT_TARGET_MODULES,
    format_e5_passage,
)


def _get_lora_target_modules(layer_indices: list[int]) -> list[str]:
    target_modules = []
    for layer_idx in layer_indices:
        for module_name in LORA_TEXT_TARGET_MODULES:
            target_modules.append(f"encoder.layer.{layer_idx}.{module_name}")
    return target_modules


def _print_trainable_params_per_layer(model: nn.Module) -> None:
    """Print number of trainable parameters grouped by layer (e.g. layer.9, head)."""
    layer_pattern = re.compile(r"encoder\.layer\.(\d+)")
    per_layer: dict[str, int] = {}
    total_trainable = 0
    total_frozen = 0
    for name, p in model.named_parameters():
        n = p.numel()
        if p.requires_grad:
            total_trainable += n
            if "head" in name and "backbone" not in name:
                key = "head"
            else:
                m = layer_pattern.search(name)
                key = f"layer.{int(m.group(1))}" if m else "other"
            per_layer[key] = per_layer.get(key, 0) + n
        else:
            total_frozen += n

    def sort_key(k: str) -> tuple[int, int]:
        if k.startswith("layer."):
            return (0, int(k.split(".")[1]))
        if k == "head":
            return (1, 0)
        return (2, 0)

    print("[Trainable parameters per layer (E5)]")
    for key in sorted(per_layer.keys(), key=sort_key):
        print(f"  {key}: {per_layer[key]:,}")
    print(f"  total trainable: {total_trainable:,}  (frozen: {total_frozen:,})")


class TextLabelDataset(Dataset):
    """Dataset of (texts, labels) for Trainer. Labels are always class indices (long): from label_encoder for classification, or bin indices for regression."""

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        label_encoder: Optional[LabelEncoder] = None,
        prefix: str = E5_PASSAGE_PREFIX,
        max_length: int = 512,
    ):
        self.texts = [str(t).strip() or " " for t in texts]
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_length = max_length
        if label_encoder is not None:
            enc = label_encoder.transform(labels.ravel().astype(str))
        else:
            enc = labels.ravel().astype(int)
        self.labels = torch.tensor(enc, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.prefix + self.texts[idx]
        out = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": out["input_ids"].squeeze(0),
            "attention_mask": out["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }


class E5ForTuning(nn.Module):
    """E5 backbone + head for classification. Same architecture always: d_output = number of unique labels (classes or bins). CE loss."""

    def __init__(
        self,
        backbone: Union[BertModel, nn.Module],
        d_output: int,
        d_model: int = E5_DIM[E5_SMALL_V2],
    ):
        super().__init__()
        self.d_output = d_output
        self.backbone = backbone
        self.head = nn.Linear(d_model, d_output)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        backbone_kwargs = {k: v for k, v in kwargs.items() if k != "num_items_in_batch"}
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **backbone_kwargs)
        last_hidden = outputs.last_hidden_state
        # TODO: test whether l2_normalize=True here improves performance — at inference embeddings
        # are normalized (see encode_texts_with_e5), so normalizing here too would be consistent
        # with E5's pre-training regime. Skipped for now as the impact is likely small.
        pooled = _mean_pool(last_hidden, attention_mask, l2_normalize=False)
        logits = self.head(pooled)
        if labels is not None:
            loss = encoder_finetune_loss(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def _load_fresh_e5(device: torch.device, model_name: str = E5_SMALL_V2) -> Tuple[Union[BertModel, nn.Module], AutoTokenizer]:
    """Load a fresh E5 model and tokenizer. Each call returns a new independent instance, so LoRA weights can be applied without affecting other uses."""
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer


_VANILLA_E5_CACHE: Dict[Tuple, Tuple[Union[BertModel, nn.Module], AutoTokenizer]] = {}


def get_vanilla_e5(device: Union[torch.device, str], model_name: str = E5_SMALL_V2) -> Tuple[Union[BertModel, nn.Module], AutoTokenizer]:
    """Return cached vanilla E5 model and tokenizer for encoding (no tuning). Same model as finetune path."""
    cache_key = (str(device), model_name)
    if cache_key not in _VANILLA_E5_CACHE:
        _VANILLA_E5_CACHE[cache_key] = _load_fresh_e5(device if isinstance(device, torch.device) else torch.device(device), model_name=model_name)
    return _VANILLA_E5_CACHE[cache_key]


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor, l2_normalize: bool = False) -> torch.Tensor:
    """Masked mean pooling over tokens. Optionally L2 normalize (for encoding; matches E5 / SentenceTransformer)."""
    mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)   # (B, 1)
    pooled = (last_hidden * mask).sum(dim=1) / sum_mask  # (B, D)
    return torch.nn.functional.normalize(pooled, p=2, dim=1) if l2_normalize else pooled


def _mean_pool_l2(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over tokens then L2 normalize (matches E5 / SentenceTransformer convention)."""
    return _mean_pool(last_hidden, attention_mask, l2_normalize=True)


def encode_texts_with_e5(
    texts: List[str],
    col_name: str,
    model: Union[BertModel, nn.Module],
    tokenizer: AutoTokenizer,
    device: Union[torch.device, str],
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    """Encode texts with a tuned or vanilla E5 model. Uses passage: col_name: col_val format when col_name is provided, else passage: text. Returns (N, D_E5)."""
    model.to(device)
    model.eval()
    prefixed = [format_e5_passage(col_name, col_val=text) for text in texts]
    embeddings_list: List[np.ndarray] = []
    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i : i + batch_size]
        out = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = out["input_ids"].to(device)
        attention_mask = out["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs["last_hidden_state"] if isinstance(outputs, dict) else outputs.last_hidden_state
        if last_hidden.size(1) == 1:
            pooled = torch.nn.functional.normalize(last_hidden.squeeze(1), p=2, dim=1)
        else:
            pooled = _mean_pool_l2(last_hidden, attention_mask)
        embeddings_list.append(pooled.cpu().numpy())
    return np.concatenate(embeddings_list, axis=0).astype(np.float32)


def finetune_e5_with_lora(
    train_texts: List[str],
    train_y: np.ndarray,
    val_texts: List[str],
    val_y: np.ndarray,
    device: Union[torch.device, str],
    tokenizer: AutoTokenizer,
    *,
    model_name: str = E5_SMALL_V2,
    lora_rank: int = 16,
    text_layers: int = 3,
    learning_rate: float = 1e-4,
    epochs: int = 40,
    patience: int = 3,
    batch_size: int = 32,
    weight_decay: float = 0.01,
    dataloader_num_workers: int = 4,
    max_length: int = 512,
    seed: int = SEED,
) -> Tuple[Union[BertModel, nn.Module], AutoTokenizer]:
    """
    Finetune an E5 model with LoRA using Hugging Face Trainer. Uses train/val for early stopping.
    Single GPU only. Returns the backbone (encoding outputs d_model dims).
    """
    assert "CUDA_VISIBLE_DEVICES" in os.environ, "CUDA_VISIBLE_DEVICES must be set"
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_model, num_layers = E5_DIM[model_name], E5_NUM_LAYERS[model_name]
    n_train, n_val = len(train_texts), len(val_texts)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    print(f"[Dataset] train: n={n_train}, val: n={n_val}")
    print(f"[Dataset] train_y.shape={train_y.shape}, val_y.shape={val_y.shape} (y = labels)")
    print(f"[Steps] batch_size={batch_size} → {steps_per_epoch} steps/epoch → {total_steps} total steps ({epochs} epochs)")

    e5_model, _ = _load_fresh_e5(device, model_name=model_name)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_y.ravel().astype(str))
    d_output_inferred = len(label_encoder.classes_)
    train_ds = TextLabelDataset(train_texts, train_y, tokenizer, label_encoder, max_length=max_length)
    val_ds = TextLabelDataset(val_texts, val_y, tokenizer, label_encoder, max_length=max_length)

    n_layers = min(max(1, text_layers), num_layers)
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
    e5_model = get_peft_model(e5_model, lora_config)
    model = E5ForTuning(backbone=e5_model, d_output=d_output_inferred, d_model=d_model)
    _print_trainable_params_per_layer(model)

    with tempfile.TemporaryDirectory(prefix="e5_finetune_") as tmpdir:
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
        def compute_metrics(eval_pred):
            return compute_metrics_multiclass(eval_pred, d_output=d_output_inferred)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )
        trainer.train()
        backbone = trainer.model.backbone

    backbone.eval()
    return backbone, tokenizer
