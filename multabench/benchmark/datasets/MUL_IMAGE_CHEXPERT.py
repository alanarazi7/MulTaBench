"""
Source: ashery/chexpert on Kaggle
        CheXpert chest X-ray dataset; predict Cardiomegaly label (0, 1, -1).

Target: Cardiomegaly (multiclass).
Features: Sex, Age, Frontal/Lateral, AP/PA, pathology labels + X-ray image.
Dropped: Path (replaced by X-Ray Image).

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


DATASET_ID = "MUL_IMAGE_CHEXPERT"
SLUG_BASE = "multabench-chexpert"
KAGGLE_SOURCE = "ashery/chexpert"

TARGET_COL = "Cardiomegaly"
IMAGE_COL = "X-Ray Image"
IMAGE_SUBFOLDER = ""

_PREFIX = "CheXpert-v1.0-small/"


def _get_img_path(path: str, dir_path: str) -> Optional[str]:
    if not path.startswith(_PREFIX):
        return None
    rel = path.replace(_PREFIX, "")
    if not exists(join(dir_path, rel)):
        return None
    return rel


def _load_and_process(dir_path: str) -> pd.DataFrame:
    train = pd.read_csv(join(dir_path, "train.csv"))
    valid = pd.read_csv(join(dir_path, "valid.csv"))
    df = pd.concat([train, valid], ignore_index=True)
    df[IMAGE_COL] = df["Path"].apply(lambda p: _get_img_path(p, dir_path))
    df = df.drop(columns=["Path"])
    df = df[df[IMAGE_COL].notna()].reset_index(drop=True)
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=dir_path,
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug,
                 image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    _args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID}")
    curate(output_dir=_args.output_dir, slug=_args.slug)
