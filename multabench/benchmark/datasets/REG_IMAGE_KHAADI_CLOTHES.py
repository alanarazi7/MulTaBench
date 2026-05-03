"""
Source: usman8/khaadis-clothes-data-with-images on Kaggle
        Khaadi clothing items with prices and product images.

Target: Price (regression).
Features: Product Description (text) + image.
Dropped: ID, Product Link, img path directory, Availability (constant).

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata

"""

import os
import shutil
from os.path import join, exists
from typing import Optional

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_KHAADI_CLOTHES"
SLUG_BASE = "multabench-khaadi-clothes"
KAGGLE_SOURCE = "usman8/khaadis-clothes-data-with-images"

TARGET_COL = "Price"
IMAGE_COL = "img"


_MAIN_DIR = "Khaadi_Data"
_IMAGE_SUBFOLDER = f"{_MAIN_DIR}/images"
_IMG_TEMP_DIR = "Img Path"
_COLS_TO_DROP = ['ID', 'Product Link', _IMG_TEMP_DIR, 'Availability', 'img_count']


def _get_img(img_dir_entry: str, dir_path: str) -> Optional[str]:
    img_folder_path = img_dir_entry.replace("images\\", "")
    candidate = join(img_folder_path, "image_0.jpg")
    if exists(join(dir_path, _IMAGE_SUBFOLDER, candidate)):
        return candidate
    return None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    main_dir = join(dir_path, _MAIN_DIR)
    df = pd.read_csv(join(main_dir, "khaadi_data.csv"))
    df[_IMG_TEMP_DIR] = df[_IMG_TEMP_DIR].apply(lambda img: img.replace("images\\", ""))
    df['img_count'] = df[_IMG_TEMP_DIR].apply(lambda i: len(os.listdir(join(dir_path, _IMAGE_SUBFOLDER, i))))
    df[IMAGE_COL] = df[_IMG_TEMP_DIR].apply(lambda i: _get_img(i, dir_path))
    drop = [c for c in _COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop)
    df = df[df[IMAGE_COL].notna()]
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    return df


def _copy_images_subdir(df: pd.DataFrame, src_dir: str, dst_dir: str) -> pd.DataFrame:
    os.makedirs(dst_dir, exist_ok=True)
    new_paths = []
    for img_path in df[IMAGE_COL]:
        flat_name = img_path.replace("/", "_").replace(os.sep, "_")
        src = join(src_dir, img_path)
        dst = join(dst_dir, flat_name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        new_paths.append(f"{IMAGES_DIR}/{flat_name}")
    df = df.copy()
    df[IMAGE_COL] = new_paths
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = _copy_images_subdir(df, src_dir=join(dir_path, _IMAGE_SUBFOLDER), dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug,
                 image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
