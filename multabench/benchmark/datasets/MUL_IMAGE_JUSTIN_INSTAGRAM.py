"""
Dataset Name: MUL_IMAGE_JUSTIN_INSTAGRAM
====
Examples: 10319
====
URL: https://www.kaggle.com/datasets/aldiandyainf/which-justin-posted-that
====
Target Variable: username (object, 5 distinct): ['justinbieber', 'justinpjtrudeau', 'justintimberlake', 'justinlong', 'justinhartley']
====
Features:

n_hashtags (int64, 17 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
n_characters (int64, 966 distinct): ['0', '14', '10', '12', '15', '17', '16', '9', '13', '7']
n_words (int64, 292 distinct): ['0', '2', '1', '3', '4', '5', '6', '7', '8', '9']
n_emojis (int64, 15 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '10', '8']
n_mentions (int64, 19 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
is_video (bool, 2 distinct): ['0', '1']
display_picture_url (object, 10319 distinct): ['images/CbNn8XPrGZG.png', 'images/CbNnwOfLtow.png', 'images/CbNmvJErdhZ.png', 'images/CbNmjmkLVKg.png', 'images/CbLpmoQPga5.png', 'images/CbJS1icPNGu.png', 'images/CbGPVftPpIo.png', 'images/CbGFsM0Pj1D.png', 'images/CbDlxuvPHvB.png', 'images/CbDZ8vhJuiF.png']
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
