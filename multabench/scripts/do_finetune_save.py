"""
Fine-tune DINO with LoRA for a dataset and save the model to disk.
Run on a GPU server. After completion, copy the checkpoint locally.

Usage:
    python do_finetune_save.py --dataset_name MUL_IMAGE_CHEXPERT --fold 0
    python do_finetune_save.py --dataset_name REG_IMAGE_KHAADI_CLOTHES --fold 0

Output:
    results/qualitative/checkpoints/{dataset_name}_fold{fold}/
"""
import argparse
import os

import numpy as np
import pandas as pd

from multabench.constants import DEVICE as _DEVICE_STR
from multabench.datasets.downloading import download_dataset
from multabench.datasets.utils import dataset_from_name
from multabench.dino.dino_finetune import finetune_dino_with_lora, _load_fresh_dino
from multabench.dino.image_loading import load_images
from multabench.preprocessing.splits import split_to_val
from tabstar.preprocessing.splits import split_to_test
from multabench.baselines.preprocessing.sampling import subsample_dataset
from tabstar2.benchmarks.evaluate import DOWNSTREAM_EXAMPLES
from tabstar2.preprocessing.feature_types import detect_image_features

_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT_DIR = os.path.join(_HERE, "results", "qualitative", "checkpoints")


def main(dataset_name: str, fold: int, lora_rank: int, img_layers: int) -> None:
    import torch
    device = torch.device(_DEVICE_STR if _DEVICE_STR else ("mps" if torch.backends.mps.is_available() else "cpu"))

    dataset_id = dataset_from_name(dataset_name)
    dataset = download_dataset(dataset_id)

    x, y = subsample_dataset(x=dataset.x, y=dataset.y, is_cls=dataset.is_cls,
                              train_examples=DOWNSTREAM_EXAMPLES, fold=fold)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=dataset.is_cls,
                                                       fold=fold, train_examples=DOWNSTREAM_EXAMPLES)
    x_tr, x_val, y_tr, y_val = split_to_val(x=x_train, y=y_train, is_cls=dataset.is_cls)

    image_cols = detect_image_features(x_train)
    assert image_cols, f"No image columns in {dataset_name}"
    img_col = image_cols[0]

    _, processor = _load_fresh_dino(device)

    print(f"Loading images for {dataset_name}...")
    train_images = load_images(s=x_tr[img_col], image_folder=dataset.image_folder)
    val_images = load_images(s=x_val[img_col], image_folder=dataset.image_folder)

    save_path = os.path.join(_CKPT_DIR, f"{dataset_name}_fold{fold}")
    os.makedirs(save_path, exist_ok=True)

    train_y = np.array(y_tr)
    val_y = np.array(y_val)
    if not dataset.is_cls:
        bins = pd.qcut(train_y, q=10, labels=False, duplicates="drop")
        bin_edges = pd.qcut(train_y, q=10, retbins=True, duplicates="drop")[1]
        train_y = bins.astype(int)
        val_y = np.clip(np.digitize(val_y, bin_edges[1:-1]), 0, len(bin_edges) - 2).astype(int)

    print(f"Fine-tuning DINO on {dataset_name}, fold {fold}...")
    finetune_dino_with_lora(
        train_images=train_images,
        train_y=train_y,
        val_images=val_images,
        val_y=val_y,
        device=device,
        processor=processor,
        lora_rank=lora_rank,
        img_layers=img_layers,
        save_path=save_path,
    )
    print(f"Done. Checkpoint at: {save_path}")
    print(f"\nTo copy locally, run:")
    print(f"  rsync -av <server>:~/multimodal-tfms/results/qualitative/checkpoints/ "
          f"./results/qualitative/checkpoints/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--img_layers", type=int, default=3)
    args = parser.parse_args()
    main(args.dataset_name, args.fold, args.lora_rank, args.img_layers)
