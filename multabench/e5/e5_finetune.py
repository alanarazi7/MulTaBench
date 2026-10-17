"""E5 text encoding, frozen or fine-tuned with LoRA on the task's labels."""
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pandas import DataFrame, Series
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from multabench.e5.constants import E5_PASSAGE_PREFIX, E5_SMALL_V2, LORA_TEXT_TARGET_MODULES, format_e5_passage
from multabench.finetune.lora import E5_LORA, LoraArgs, TunableEncoder, finetune_with_lora, finetuning_split
from multabench.utils.timing import embedding_step

MAX_LENGTH = 512
ENCODE_BATCH_SIZE = 32

Device = Union[torch.device, str]
E5 = Tuple[nn.Module, PreTrainedTokenizerBase]

_VANILLA_E5: Dict[Tuple[str, str], E5] = {}


def load_e5(device: Device, model_name: str = E5_SMALL_V2) -> E5:
    """A fresh model, so LoRA can adapt it without touching the shared frozen one."""
    return AutoModel.from_pretrained(model_name).to(device), AutoTokenizer.from_pretrained(model_name)


@embedding_step
def get_vanilla_e5(device: Device, model_name: str = E5_SMALL_V2) -> E5:
    key = (str(device), model_name)
    if key not in _VANILLA_E5:
        _VANILLA_E5[key] = load_e5(device, model_name=model_name)
    return _VANILLA_E5[key]


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor, l2_normalize: bool = False) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1) if l2_normalize else pooled


@embedding_step
def encode_texts_with_e5(texts: Sequence[str], col_name: str, model: nn.Module, tokenizer: PreTrainedTokenizerBase,
                         device: Device) -> np.ndarray:
    """L2-normalized mean-pooled embeddings of "passage: <col_name>: <text>", as E5 is trained; (N, d_model)."""
    model.to(device)
    model.eval()
    passages = [format_e5_passage(col_name, col_val=text) for text in texts]
    embeddings = []
    for i in range(0, len(passages), ENCODE_BATCH_SIZE):
        inputs = tokenizer(passages[i:i + ENCODE_BATCH_SIZE], padding=True, truncation=True, max_length=MAX_LENGTH,
                           return_tensors="pt").to(device)
        mask = inputs["attention_mask"]
        with torch.no_grad():
            last_hidden = model(input_ids=inputs["input_ids"], attention_mask=mask).last_hidden_state
        embeddings.append(mean_pool(last_hidden, mask, l2_normalize=True).cpu().numpy())
    return np.concatenate(embeddings).astype(np.float32)


def finetune_e5(x: DataFrame, y: Series, text_features: List[str], is_cls: bool, device: Device,
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
                        max_length=MAX_LENGTH, return_tensors="pt")
        return {"input_ids": out["input_ids"].squeeze(0), "attention_mask": out["attention_mask"].squeeze(0)}

    encoder = TunableEncoder(
        load_backbone=lambda: load_e5(device, model_name=model_name)[0],
        layer_modules=lambda layer: [f"encoder.layer.{layer}.{m}" for m in LORA_TEXT_TARGET_MODULES],
        featurize=featurize,
        pool=lambda last_hidden, inputs: mean_pool(last_hidden, inputs["attention_mask"]))
    print(f"Fine-tuning one {model_name} for the text columns {text_features}")
    backbone, stats = finetune_with_lora(encoder, *examples(x_tr, y_tr), *examples(x_val, y_val), args=args)
    return backbone.to(device), tokenizer, stats
