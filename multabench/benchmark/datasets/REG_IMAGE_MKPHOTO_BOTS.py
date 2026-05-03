"""
Source: guardeec/mkphoto2023 on Kaggle
        Profile photos of malicious VK social bots; predict trust score (regression).

Target: TrustNZ_IDNNZ (regression).
Features: gan, dtm, person, n_faces, celebs, BTT, speed, NBQ + face image.
Dropped: id, dataset (leakage), SR and price (too easy).

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata

"""

import os
from os.path import join, exists
from typing import Optional

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_MKPHOTO_BOTS"
SLUG_BASE = "multabench-mkphoto-bots"
KAGGLE_SOURCE = "guardeec/mkphoto2023"

TARGET_COL = "TrustNZ_IDNNZ"
IMAGE_COL = "image"
IMAGE_SUBFOLDER = "faces_400/faces_400"

_COLS_TO_DROP = ["id", "dataset", "SR", "price"]


def _get_img(img_id: str, dir_path: str) -> Optional[str]:
    img_name = f"{img_id}.jpg"
    if not exists(join(dir_path, IMAGE_SUBFOLDER, img_name)):
        return None
    return img_name


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "dataset.csv"))
    df[IMAGE_COL] = df["id"].apply(lambda i: _get_img(i, dir_path))
    df = df[df[IMAGE_COL].notna()].reset_index(drop=True)
    drop = [c for c in _COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop)
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER),
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug,
                 image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    _args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID}")
    curate(output_dir=_args.output_dir, slug=_args.slug)
