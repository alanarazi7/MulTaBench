"""Shared utilities for leaderboard display modules."""
import matplotlib.pyplot as plt
from multabench.leaderboard.data.keys import MODALITY_IMAGE, MODALITY_TEXT, TASK_REG, TASK_CLS


def save_figure(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, format="pdf", dpi=200, bbox_inches="tight", pad_inches=0)


def infer_modality(dataset_name: str) -> str:
    parts = dataset_name.split("_")
    if "TEXT" in parts:
        return MODALITY_TEXT
    if "IMAGE" in parts:
        return MODALITY_IMAGE
    return "Other"


def infer_task_type(dataset_name: str) -> str:
    return TASK_REG if dataset_name.startswith("REG_") else TASK_CLS


def badge(count: int, total: int = 5) -> str:
    emoji = "✅" if count >= total else ("⚠️" if count > total / 2 else "❌")
    return f"{emoji} {count}/{total}"
