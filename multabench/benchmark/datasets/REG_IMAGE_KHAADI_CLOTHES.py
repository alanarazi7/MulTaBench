"""
Dataset Name: REG_IMAGE_KHAADI_CLOTHES
====
Examples: 400
====
URL: https://www.kaggle.com/datasets/usman8/khaadis-clothes-data-with-images
====
Target Variable: Price (int64, 55 distinct): ['3490', '3990', '2490', '3690', '4990', '4190', '1990', '3190', '2990', '5990']
====
Features:

Product Name (object, 83 distinct): ['Fabrics 3 Piece Suit', 'Fabrics 2 Piece', 'Classic Kameez', 'Classic Kurta', 'Narrow Culottes', 'Dupatta', 'Button Down Shirt', 'Flared Kameez', 'Drop Shoulder', 'Contemporary Kameez']
Product Description (object, 128 distinct): ['Printed | Cambric', 'Dyed Embroidered | Viscose Oak Silk', 'Yarn Dyed Embroidered | Cotton Net', 'Printed | Viscose Crepe', 'Printed | Lawn', 'Printed Lawn | Top Dupatta', 'Dyed Embroidered | Dobby', 'Yarn Dyed | Cotton Polyester Broshia Jacquard', 'Dyed Embroidered | Dull Raw Silk', 'Dyed Embroidered | Viscose Crepe']
Color (object, 29 distinct): ['BLACK', 'BLUE', 'MULTI', 'GREEN', 'OFF-WHITE', 'PINK', 'BEIGE', 'RED', 'WHITE', 'PURPLE']
img (object, 400 distinct): ['images/ALK231009_image_0.jpg', 'images/BLK231004_image_0.jpg', 'images/ACA231008_image_0.jpg', 'images/ILK231001_image_0.jpg', 'images/JK231001_image_0.jpg', 'images/MLK231001_image_0.jpg', 'images/AK231006_image_0.jpg', 'images/BLK231001_image_0.jpg', 'images/JK231002_image_0.jpg', 'images/BCH231002_image_0.jpg']
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
