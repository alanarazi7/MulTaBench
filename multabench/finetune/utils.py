"""Shared utilities for E5/DINO finetuning."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def encoder_finetune_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Classification loss for E5/DINO encoder finetuning. Always cross-entropy over classes or bins.
    Regression is handled by binning upstream; we never use MSE here.
    Flattens labels to (B,) so (B, 1) or (B, 1, 1) from collators do not cause shape mismatches.
    """
    labels_flat = labels.view(-1).long()
    return torch.nn.functional.cross_entropy(logits, labels_flat)


def compute_metrics_multiclass(eval_pred: Any, d_output: int) -> Dict[str, float]:
    """Eval metric for encoder finetuning: always multiclass (classes or bins). Returns eval_auc."""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    probs = torch.softmax(torch.tensor(predictions), dim=-1).numpy()
    labels_flat = np.asarray(labels).reshape(-1)
    if d_output == 2:
        score = roc_auc_score(labels_flat, probs[:, 1])
    else:
        try:
            score = roc_auc_score(labels_flat, probs, multi_class="ovr", average="macro")
        except ValueError:
            score = _per_class_auc(labels_flat, probs)
    return {"eval_auc": float(score)}


def _per_class_auc(y_true, y_pred) -> float:
    present_classes = np.unique(y_true)
    aucs = {}
    for cls in present_classes:
        # Binary ground truth: 1 for the current class, 0 for others
        y_true_binary = (y_true == cls).astype(int)
        # Predicted probabilities for the current class
        y_pred_scores = y_pred[:, int(cls)]
        try:
            auc = roc_auc_score(y_true_binary, y_pred_scores)
            aucs[cls] = auc
        except ValueError:
            pass
    macro_avg = float(np.mean(list(aucs.values())))
    return macro_avg
