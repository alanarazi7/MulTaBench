"""
Curation script for REG_IMAGE_PAINTING_PRICE_PREDICTION.

Source: denozavrus/paintings-price-prediction on Kaggle
        12,369 paintings with images and rich metadata scraped from an online
        art marketplace.

Target: price — listing price in USD (regression).
Features: material, styles, width, length + ~90 binary style/genre tag columns + image.
Dropped: price=0 rows (unlisted / free works with no meaningful price signal).

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


DATASET_ID = "REG_IMAGE_PAINTING_PRICE"
SLUG_BASE = "multabench-painting-price"
KAGGLE_SOURCE = "denozavrus/paintings-price-prediction"

TARGET_COL = "price"
IMAGE_COL = "image_url"
IMAGE_SUBFOLDER = "images/images"



def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "paintings_data.csv"))
    df = df.dropna(subset=[IMAGE_COL])
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER), dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug,
                 image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
