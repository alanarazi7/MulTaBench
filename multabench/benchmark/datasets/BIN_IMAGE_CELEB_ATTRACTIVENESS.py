"""
Source: jessicali9530/celeba-dataset on Kaggle
        Over 200K celebrity face images with 40 binary attribute annotations.

Samples 100K rows from 202,599 total. Target: Attractive (-1/1).
Features: 39 binary facial attributes + image_id (image).

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


DATASET_ID = "BIN_IMAGE_CELEB_ATTRACTIVENESS"
SLUG_BASE = "multabench-celeb-attractiveness"
KAGGLE_SOURCE = "jessicali9530/celeba-dataset"

TARGET_COL = "Attractive"
IMAGE_COL = "image_id"
IMAGE_SUBFOLDER = "img_align_celeba/img_align_celeba"
SAMPLE_N = 100_000
RANDOM_STATE = 42



def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "list_attr_celeba.csv"))
    df = df.sample(n=SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER), dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID), kaggle_source=KAGGLE_SOURCE)


