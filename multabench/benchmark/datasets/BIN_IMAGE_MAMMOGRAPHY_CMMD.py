"""
Source: nguynththanhho/cmmd-mammography on Kaggle
        Chinese Mammography (CMMD) breast cancer images; binary benign/malignant classification.

Target: Classification (binary).
Features: Age, LeftRight, Abnormality, Method_crop + image.
Dropped: ID, Subtype (43% missing).

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata

"""

import os
from os.path import join, exists

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "BIN_IMAGE_MAMMOGRAPHY_CMMD"
SLUG_BASE = "multabench-mammography-cmmd"
KAGGLE_SOURCE = "nguynththanhho/cmmd-mammography"

TARGET_COL = "Classification"
IMAGE_COL = "Path"
IMAGE_SUBFOLDER = "CMMD"

_COLS_TO_DROP = ["ID", "Subtype"]


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "CMMD", "description.csv"))
    df[IMAGE_COL] = df[IMAGE_COL].apply(
        lambda p: p if exists(join(dir_path, IMAGE_SUBFOLDER, p)) else None
    )
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
