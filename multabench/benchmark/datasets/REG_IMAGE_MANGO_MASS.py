"""
Source: mypapit/mangomassnet552-dataset on Kaggle
        Harumanis mango images with mass (weight) labels.

Target: Mass(kg) (regression).
Features: image only.

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


DATASET_ID = "REG_IMAGE_MANGO_MASS"
SLUG_BASE = "multabench-mango-mass"
KAGGLE_SOURCE = "mypapit/mangomassnet552-dataset"

TARGET_COL = "Mass(kg)"
IMAGE_COL = "Fruit No"
IMAGE_SUBFOLDER = "images"



def _remove_missing(img: str, dir_path: str) -> Optional[str]:
    if not exists(join(dir_path, IMAGE_SUBFOLDER, img)):
        return None
    return img


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_excel(join(dir_path, "Harumanis_mango_weight_grade.xlsx"))
    df[IMAGE_COL] = df[IMAGE_COL].apply(lambda img: _remove_missing(img, dir_path))
    df = df[df[IMAGE_COL].notna()]
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
