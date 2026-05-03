"""
Visualise frozen encoder attention for a single dataset.
Phase 1 of qualitative analysis -- no fine-tuning involved.

Usage:
    python do_attention.py --dataset_name REG_IMAGE_KHAADI_CLOTHES --fold 0
    python do_attention.py --dataset_name BIN_TEXT_JIGSAW_TOXICITY --fold 0
    python do_attention.py --dataset_name REG_IMAGE_KHAADI_CLOTHES --fold 0 --n_examples 3
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from multabench.constants import DEVICE as _DEVICE_STR
from multabench.datasets.all_datasets import is_image_dataset
from multabench.datasets.downloading import download_dataset
from multabench.datasets.utils import dataset_from_name
from multabench.dino.dino_finetune import _load_fresh_dino
from multabench.dino.image_loading import load_images
from multabench.e5.constants import format_e5_passage
from multabench.e5.e5_finetune import get_vanilla_e5
from multabench.preprocessing.feat_types import detect_text_features
from tabstar.preprocessing.splits import split_to_test
from multabench.baselines.preprocessing.sampling import subsample_dataset
from tabstar2.benchmarks.evaluate import DOWNSTREAM_EXAMPLES
from tabstar2.preprocessing.feature_types import detect_image_features

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_HERE, "results", "qualitative")
_TILES_DIR = os.path.join(_OUT_DIR, "tiles")
_DEVICE = torch.device(_DEVICE_STR if _DEVICE_STR else ("mps" if torch.backends.mps.is_available() else "cpu"))


def _pick_examples(y, n_per_class: int, is_cls: bool) -> tuple[list[int], list]:
    """Return (row_indices, row_labels) with up to n_per_class rows per class, sorted by class."""
    y_arr = np.array(y)
    if is_cls:
        idxs, labels = [], []
        for cls in np.unique(y_arr):
            pool = list(np.where(y_arr == cls)[0])
            for pos in pool[:n_per_class]:
                idxs.append(int(pos))
                labels.append(cls)
    else:
        quantiles = np.linspace(0, 1, n_per_class)
        thresholds = np.quantile(y_arr, quantiles)
        idxs = [int(np.argmin(np.abs(y_arr - t))) for t in thresholds]
        labels = [y_arr[i] for i in idxs]
    return idxs, labels


def _attn_overlay(img: Image.Image, attn_196: np.ndarray) -> np.ndarray:
    """Blend jet attention heatmap onto image. Returns (224, 224, 3) uint8."""
    img_224 = img.convert("RGB").resize((224, 224))
    orig = np.array(img_224, dtype=np.float32) / 255.0
    n_side = int(len(attn_196) ** 0.5)
    attn = torch.from_numpy(attn_196.reshape(1, 1, n_side, n_side).astype(np.float32))
    attn_up = torch.nn.functional.interpolate(attn, size=(224, 224), mode="bicubic", align_corners=False)[0, 0].numpy()
    attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
    heatmap = cm.jet(attn_norm)[:, :, :3]
    overlay = 0.55 * orig + 0.45 * heatmap
    return (overlay.clip(0, 1) * 255).astype(np.uint8)


def _load_and_split(dataset_name: str, fold: int):
    dataset_id = dataset_from_name(dataset_name)
    dataset = download_dataset(dataset_id)
    x, y = subsample_dataset(x=dataset.x, y=dataset.y, is_cls=dataset.is_cls,
                              train_examples=DOWNSTREAM_EXAMPLES, fold=fold)
    _, x_test, _, y_test = split_to_test(x=x, y=y, is_cls=dataset.is_cls,
                                         fold=fold, train_examples=DOWNSTREAM_EXAMPLES)
    return dataset, x_test, y_test


def _extract_attn_maps(model, processor, images: list, device) -> list:
    maps = []
    for img in images:
        inputs = processor(images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        last_attn = outputs.attentions[-1][0]
        all_attn = last_attn[:, 0, 1:].mean(0).cpu().numpy()
        n_side = int(len(all_attn) ** 0.5)
        maps.append(all_attn[-(n_side * n_side):])
    return maps


def run_image_attention(dataset_name: str, fold: int, n_per_class: int,
                        checkpoint: str | None = None,
                        only_idxs: list[int] | None = None) -> None:
    from transformers import AutoModel, AutoImageProcessor
    device = _DEVICE
    dataset, x_test, y_test = _load_and_split(dataset_name, fold)

    image_cols = detect_image_features(x_test)
    assert image_cols, f"No image columns found in {dataset_name}"
    img_col = image_cols[0]

    if only_idxs is not None:
        # User explicitly picked test-set indices; respect order exactly
        row_idxs = only_idxs
        row_labels = [y_test.iloc[i] for i in row_idxs]
    else:
        # Deterministic: sorted by class, then by position in test set
        row_idxs, row_labels = _pick_examples(y_test, n_per_class=n_per_class, is_cls=dataset.is_cls)

    example_paths = x_test.iloc[row_idxs][img_col].tolist()
    example_images = load_images(s=example_paths, image_folder=dataset.image_folder)
    print(f"Selected {len(row_idxs)} examples (test-set indices): {row_idxs}")

    frozen_model, processor = _load_fresh_dino(device)
    frozen_model.set_attn_implementation("eager")
    frozen_model.eval()
    frozen_maps = _extract_attn_maps(frozen_model, processor, example_images, device)
    del frozen_model

    ft_maps = None
    if checkpoint is not None:
        ft_model = AutoModel.from_pretrained(checkpoint).to(device)
        ft_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
        ft_model.set_attn_implementation("eager")
        ft_model.eval()
        ft_maps = _extract_attn_maps(ft_model, ft_processor, example_images, device)

    n_actual = len(example_images)
    col_specs = [("original", None), ("frozen", frozen_maps)]
    if ft_maps is not None:
        col_specs.append(("contextualized", ft_maps))
    n_cols = len(col_specs)

    # Save individual tiles when working with explicit indices
    if only_idxs is not None:
        tile_dir = os.path.join(_TILES_DIR, f"{dataset_name}_fold{fold}")
        os.makedirs(tile_dir, exist_ok=True)
        for i, (img, target, test_idx) in enumerate(zip(example_images, row_labels, row_idxs)):
            label_str = f"{target:.3g}" if isinstance(target, float) else str(target)
            label_str = label_str.replace("-", "neg").replace(" ", "_").replace("%", "pct")
            for col_name, maps in col_specs:
                tile_path = os.path.join(tile_dir, f"idx{test_idx}_label{label_str}_{col_name}.png")
                if maps is None:
                    img.convert("RGB").resize((224, 224)).save(tile_path)
                else:
                    Image.fromarray(_attn_overlay(img, maps[i])).save(tile_path)
                print(f"  Tile → {tile_path}")

    col_titles = {"original": "Image", "frozen": "Frozen", "contextualized": "Contextualized"}
    fig, axes = plt.subplots(n_actual, n_cols, figsize=(3 * n_cols, 3.2 * n_actual))
    if n_actual == 1:
        axes = [axes]
    for i, (img, target, test_idx) in enumerate(zip(example_images, row_labels, row_idxs)):
        label = f"{target:.3g}" if isinstance(target, float) else str(target)
        for j, (col_name, maps) in enumerate(col_specs):
            ax = axes[i][j]
            ax.axis("off")
            if maps is None:
                ax.imshow(img.convert("RGB").resize((224, 224)))
                ax.set_title(f"idx={test_idx}  label={label}", fontsize=10, fontweight="bold")
            else:
                ax.imshow(_attn_overlay(img, maps[i]))
                ax.set_title(col_titles[col_name], fontsize=9)

    suffix = "comparison" if ft_maps is not None else "frozen_attn"
    fig.suptitle(f"{dataset_name} | fold {fold}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, dataset_name, fold, suffix=suffix)


def run_text_attention(dataset_name: str, fold: int, n_examples: int) -> None:
    device = _DEVICE
    dataset, x_test, y_test = _load_and_split(dataset_name, fold)

    text_cols = detect_text_features(x_test)
    assert text_cols, f"No text columns found in {dataset_name}"
    text_col = text_cols[0]

    idxs, _ = _pick_examples(y_test, n_per_class=n_examples, is_cls=dataset.is_cls)
    example_texts = x_test.iloc[idxs][text_col].astype(str).tolist()
    example_targets = y_test.iloc[idxs].tolist()

    model, tokenizer = get_vanilla_e5(device)
    model.set_attn_implementation("eager")
    model.eval()

    n_actual = len(example_texts)
    fig, axes = plt.subplots(n_actual, 1, figsize=(14, 3 * n_actual))
    if n_actual == 1:
        axes = [axes]

    for i, (text, target) in enumerate(zip(example_texts, example_targets)):
        prefixed = format_e5_passage(col_name=text_col, col_val=text)
        enc = tokenizer(prefixed, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            outputs = model(**enc, output_attentions=True)
        last_attn = outputs.attentions[-1][0]       # (n_heads, seq_len, seq_len)
        cls_attn = last_attn[:, 0, :].mean(0)       # (seq_len,)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        scores = cls_attn.cpu().numpy()

        keep = [j for j, t in enumerate(tokens) if t not in ("[CLS]", "[SEP]", "[PAD]")]
        tokens_k = [tokens[j] for j in keep]
        scores_k = scores[keep]

        ax = axes[i]
        ax.bar(range(len(tokens_k)), scores_k, color="#A8D4F0", edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(tokens_k)))
        ax.set_xticklabels(tokens_k, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"target={target} | {text[:80]}", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{dataset_name} (col: {text_col}) | fold {fold} | Frozen E5", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, dataset_name, fold, suffix="frozen_attn")


def _save(fig, dataset_name: str, fold: int, suffix: str = "frozen_attn") -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, f"{dataset_name}_fold{fold}_{suffix}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_per_class", type=int, default=5,
                        help="Max examples per class (classification) or total (regression)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to saved fine-tuned model for before/after comparison")
    parser.add_argument("--idxs", type=int, nargs="+", default=None,
                        help="Explicit test-set indices to visualise (overrides --n_per_class)")
    args = parser.parse_args()

    dataset_id = dataset_from_name(args.dataset_name)
    if is_image_dataset(dataset_id):
        run_image_attention(args.dataset_name, args.fold, args.n_per_class,
                            checkpoint=args.checkpoint, only_idxs=args.idxs)
    else:
        run_text_attention(args.dataset_name, args.fold, args.n_per_class)
