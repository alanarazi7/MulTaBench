"""
Source: aldiandyainf/which-justin-posted-that on Kaggle
        Instagram posts from three famous Justins, predict the poster.

Target: username (multiclass).
Features: image only.
Dropped: display_picture_relative_url, urls, n_likes, n_comments, captions, post_dates.

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


DATASET_ID = "MUL_IMAGE_JUSTIN_INSTAGRAM"
SLUG_BASE = "multabench-justin-instagram"
KAGGLE_SOURCE = "aldiandyainf/which-justin-posted-that"

TARGET_COL = "username"
IMAGE_COL = "display_picture_url"
IMAGE_SUBFOLDER = "imgs/imgs"

_IMG_RAW = "display_picture_relative_url"
_COLS_TO_DROP = [_IMG_RAW, 'urls', 'n_likes', 'n_comments', 'captions', 'post_dates']



def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "dataset.csv"))
    df[IMAGE_COL] = df[_IMG_RAW].apply(lambda img: img.split('/')[-1])
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
