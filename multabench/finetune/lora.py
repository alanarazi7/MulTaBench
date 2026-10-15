"""LoRA fine-tuning of a pretrained encoder under a linear head, shared by E5 (text) and DINO (images).

Regression targets are binned into quantiles, so the head is always a classifier trained with cross-entropy; the
Trainer early-stops on a validation split and keeps the best epoch. Only the tuned backbone is returned.
"""
import re
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from pandas import DataFrame, Series
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from multabench.constants import SEED
from multabench.finetune.utils import compute_metrics_multiclass, encoder_finetune_loss
from multabench.preprocessing.discretize import discretize_numerical
from multabench.preprocessing.splits import split_to_val
from multabench.utils.timing import embedding_step

REGRESSION_BINS = 20
DATALOADER_WORKERS = 4

Pool = Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]


@dataclass(frozen=True)
class LoraArgs:
    learning_rate: float
    epochs: int
    lora_rank: int = 16
    layers: int = 3
    patience: int = 3
    weight_decay: float = 0.001
    batch_size: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


E5_LORA = LoraArgs(learning_rate=1e-4, epochs=50)
DINO_LORA = LoraArgs(learning_rate=1e-3, epochs=100)


@dataclass(frozen=True)
class TunableEncoder:
    """What fine-tuning needs of an encoder: a fresh backbone, the modules LoRA adapts in a layer, how one input
    becomes the backbone's tensors, and how its last hidden state is pooled into one vector."""
    load_backbone: Callable[[], nn.Module]
    layer_modules: Callable[[int], List[str]]
    featurize: Callable[[Any], Dict[str, torch.Tensor]]
    pool: Pool


class EncoderWithHead(nn.Module):

    def __init__(self, backbone: nn.Module, pool: Pool, d_output: int):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = nn.Linear(backbone.config.hidden_size, d_output)

    def forward(self, labels: torch.Tensor | None = None, **inputs: Any) -> Dict[str, torch.Tensor]:
        inputs.pop("num_items_in_batch", None)
        logits = self.head(self.pool(self.backbone(**inputs).last_hidden_state, inputs))
        if labels is None:
            return {"logits": logits}
        return {"loss": encoder_finetune_loss(logits, labels), "logits": logits}


class LabeledDataset(Dataset):

    def __init__(self, items: Sequence, labels: np.ndarray, label_encoder: LabelEncoder,
                 featurize: Callable[[Any], Dict[str, torch.Tensor]]):
        self.items = items
        self.labels = torch.tensor(label_encoder.transform(labels.ravel().astype(str)), dtype=torch.long)
        self.featurize = featurize

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {**self.featurize(self.items[idx]), "labels": self.labels[idx]}


def finetuning_split(x: DataFrame, y: Series, is_cls: bool) -> Tuple[DataFrame, DataFrame, Series, Series]:
    """The training rows split into train and validation, with regression targets binned into quantiles."""
    if not is_cls:
        y = discretize_numerical(y, n_bins=REGRESSION_BINS)
    return split_to_val(x=x, y=y, is_cls=True)


@embedding_step
def finetune_with_lora(encoder: TunableEncoder, train_items: Sequence, train_y: np.ndarray, val_items: Sequence,
                       val_y: np.ndarray, args: LoraArgs, seed: int = SEED) -> Tuple[nn.Module, dict]:
    """LoRA on the last args.layers layers: the backbone of the epoch with the best validation loss, and its stats."""
    assert torch.cuda.device_count() <= 1, "The Trainer uses every visible GPU; make only one visible"
    torch.manual_seed(seed)
    np.random.seed(seed)
    backbone = encoder.load_backbone()
    label_encoder = LabelEncoder().fit(train_y.ravel().astype(str))
    d_output = len(label_encoder.classes_)
    train_ds = LabeledDataset(train_items, train_y, label_encoder, encoder.featurize)
    val_ds = LabeledDataset(val_items, val_y, label_encoder, encoder.featurize)
    num_layers = backbone.config.num_hidden_layers
    n_layers = min(max(1, args.layers), num_layers)
    print(f"🦜 LoRA rank {args.lora_rank} on the last {n_layers} layers, {len(train_ds)} train and {len(val_ds)} "
          f"validation examples, {d_output} classes")
    targets = [module for layer in range(num_layers - n_layers, num_layers) for module in encoder.layer_modules(layer)]
    config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2, target_modules=targets, lora_dropout=0.1,
                        bias="none")
    model = EncoderWithHead(get_peft_model(backbone, config), pool=encoder.pool, d_output=d_output)
    print_trainable_params(model)
    with tempfile.TemporaryDirectory(prefix="lora_finetune_") as tmpdir:
        training_args = TrainingArguments(
            output_dir=tmpdir, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=1, seed=seed,
            remove_unused_columns=False, report_to="none", dataloader_num_workers=DATALOADER_WORKERS)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
                          compute_metrics=lambda pred: compute_metrics_multiclass(pred, d_output=d_output),
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)])
        trainer.train()
        backbone = trainer.model.backbone
    stats = {"best_val_loss": trainer.state.best_metric, "epochs": trainer.state.epoch}
    print(f"Fine-tuned for {stats['epochs']:.0f} epochs, best validation loss {stats['best_val_loss']:.4f}")
    backbone.eval()
    return backbone, stats


def print_trainable_params(model: nn.Module):
    layer_re = re.compile(r"layer\.(\d+)")
    groups: Dict[str, int] = {}
    trainable, frozen = 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            frozen += p.numel()
            continue
        trainable += p.numel()
        m = layer_re.search(name)
        key = f"layer.{int(m.group(1))}" if m else ("head" if name.startswith("head") else "other")
        groups[key] = groups.get(key, 0) + p.numel()
    for key in sorted(groups, key=lambda k: (0, int(k.split(".")[1])) if k.startswith("layer.") else (1, 0)):
        print(f"  {key}: {groups[key]:,}")
    print(f"  total trainable: {trainable:,}  (frozen: {frozen:,})")
