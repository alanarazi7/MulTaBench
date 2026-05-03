"""
Source: parthplc/facebook-hateful-meme-dataset on Kaggle
        Facebook AI Hateful Memes Challenge dataset (train + dev + test splits).

Loads all 3 jsonl splits, encodes meme text with E5-small-v2 + PCA(20),
and includes meme images.

Target: label (0=not-hateful, 1=hateful). Metric: AUROC.
Features: img (image), dim_e5_1..20 (E5 text embeddings of overlay text).

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata
        annotation.md     dataset description
"""

import os
from os.path import join

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from multabench.e5.e5_finetune import get_vanilla_e5, encode_texts_with_e5
from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name
from multabench.utils.io_handlers import load_json_lines


DATASET_ID = "BIN_IMAGE_HATEFUL_MEME"
SLUG_BASE = "multabench-hateful-meme"
KAGGLE_SOURCE = "parthplc/facebook-hateful-meme-dataset"

TARGET_COL = "label"
IMAGE_COL = "img"
DATA_SUBFOLDER = "data"
N_E5_DIMS = 20


def _load_and_process(dir_path: str) -> pd.DataFrame:
    main_path = join(dir_path, DATA_SUBFOLDER)
    rows = []
    for split in ["train", "dev", "test"]:
        split_path = join(main_path, f"{split}.jsonl")
        for d in load_json_lines(split_path):
            d.pop("id", None)
            rows.append(d)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    print(f"[hateful_meme] loaded {len(df)} rows (dropped unlabeled test rows), encoding text with E5...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_vanilla_e5(device)
    texts = df["text"].fillna("").tolist()
    embeddings = encode_texts_with_e5(texts, col_name="text", model=model, tokenizer=tokenizer, device=device)
    print(f"[hateful_meme] E5 done ({embeddings.shape}), running PCA to {N_E5_DIMS} dims...")
    pca = PCA(n_components=N_E5_DIMS, random_state=42)
    reduced = pca.fit_transform(embeddings)
    for i in range(N_E5_DIMS):
        df[f"dim_e5_{i + 1}"] = reduced[:, i]
    df = df.drop(columns=["text"])
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    # img paths in jsonl are like "img/42953.png" relative to the data/ folder
    df = copy_images(df=df, image_col=IMAGE_COL,
                     src_dir=join(dir_path, DATA_SUBFOLDER),
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL,
                 dataset_id=DATASET_ID, slug=slug, image_col=IMAGE_COL,
                 task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate Facebook Hateful Meme dataset for MulTaBench")
    curate(output_dir=args.output_dir, slug=args.slug)
