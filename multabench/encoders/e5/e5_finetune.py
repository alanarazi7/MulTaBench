"""E5: loading, encoding, and LoRA fine-tuning on (text, label) with finetune/lora.py."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pandas import DataFrame, Series
from transformers import AutoModel, AutoTokenizer, BertModel

from multabench.encoders.e5.constants import (E5_LORA, E5_PASSAGE_PREFIX, E5_SMALL_V2, LORA_TEXT_TARGET_MODULES,
                                              format_e5_passage)
from multabench.finetune.lora import TunableEncoder, finetune_with_lora, finetuning_split
from multabench.finetune.train_args import LoraArgs
from multabench.utils.timing import embedding_step


def _load_fresh_e5(device: torch.device, model_name: str = E5_SMALL_V2) -> Tuple[Union[BertModel, nn.Module], AutoTokenizer]:
    """Load a fresh E5 model and tokenizer. Each call returns a new independent instance, so LoRA weights can be applied without affecting other uses."""
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer


_VANILLA_E5_CACHE: Dict[Tuple, Tuple[Union[BertModel, nn.Module], AutoTokenizer]] = {}


@embedding_step
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


@embedding_step
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
            padding=True,
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


def finetune_e5(x: DataFrame, y: Series, text_features: List[str], is_cls: bool, device: Union[torch.device, str],
                model_name: str = E5_SMALL_V2, args: LoraArgs = E5_LORA) -> Tuple[nn.Module, Any, dict]:
    """One E5 for all text columns, tuned on "<column>: <text>" of every training row and column, each labeled with its
    row's target. x holds the texts as runs embed them, after the missing-text fill."""
    x_tr, x_val, y_tr, y_val = finetuning_split(x=x, y=y, is_cls=is_cls)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def examples(rows: DataFrame, labels: Series) -> Tuple[List[str], np.ndarray]:
        texts = [f"{col}: {str(rows.at[idx, col]).strip()}" for idx in rows.index for col in text_features]
        return texts, np.repeat(labels.loc[rows.index].to_numpy(), len(text_features))

    def featurize(text: str) -> Dict[str, torch.Tensor]:
        out = tokenizer(E5_PASSAGE_PREFIX + (text.strip() or " "), padding="max_length", truncation=True,
                        max_length=512, return_tensors="pt")
        return {"input_ids": out["input_ids"].squeeze(0), "attention_mask": out["attention_mask"].squeeze(0)}

    encoder = TunableEncoder(
        load_backbone=lambda: _load_fresh_e5(device, model_name=model_name)[0],
        layer_modules=lambda layer: [f"encoder.layer.{layer}.{m}" for m in LORA_TEXT_TARGET_MODULES],
        featurize=featurize,
        pool=lambda last_hidden, inputs: _mean_pool(last_hidden, inputs["attention_mask"]))
    print(f"Fine-tuning one {model_name} for the text columns {text_features}")
    backbone, stats = finetune_with_lora(encoder, *examples(x_tr, y_tr), *examples(x_val, y_val), args=args)
    return backbone.to(device), tokenizer, stats
