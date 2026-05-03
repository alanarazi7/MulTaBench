"""
Source: samoilovmikhail/floral-bouquets-images-and-girlfriend-scores on Kaggle
        Floral bouquet images with girlfriend taste ratings.

Target: girlfriend_rating (multiclass).
Features: description (text) + image.
Dropped: product_id.

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata

"""

import os
from os.path import join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "MUL_IMAGE_FLOWERS_BOUQUETS_GIRLFRIEND_TASTE"
SLUG_BASE = "multabench-flower-bouquets"
KAGGLE_SOURCE = "samoilovmikhail/floral-bouquets-images-and-girlfriend-scores"

TARGET_COL = "girlfriend_rating"
IMAGE_COL = "image_name"
IMAGE_SUBFOLDER = "images"



def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "flowers.csv"))
    df = df.drop(columns=["product_id"], errors="ignore")
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
