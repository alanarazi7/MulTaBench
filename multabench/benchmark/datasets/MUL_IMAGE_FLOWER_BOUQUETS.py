"""
Dataset Name: MUL_IMAGE_FLOWERS_BOUQUETS_GIRLFRIEND_TASTE
====
Examples: 600
====
URL: https://www.kaggle.com/datasets/olgabelitskaya/flower-color-images
====
Target Variable: girlfriend_rating (int64, 5 distinct): ['1', '5', '4', '2', '3']
====
Features:

image_name (object, 600 distinct): ['images/0000_Bouquet_of_5_eustoms_in_craft.jpg', 'images/0001_Bouquet_of_11_peony-shaped_bush_roses.jpg', 'images/0002_Bouquet_Lydia.jpg', 'images/0003_Bouquet_of_19_white_roses_in_a_designer_package.jpg', 'images/0004_Bouquet_of_15_alstroemeria_with_greenery_in_craft.jpg', 'images/0005_Peonies_9_pieces.jpg', 'images/0006_Sunflowers_9_pieces.jpg', 'images/0007_25_red_roses.jpg', 'images/0008_Red_Roses_Russia_21_pcs.jpg', 'images/0009_Mono_peony-shaped_bush_roses_Madame_Bambastic9_pc.jpg']
description (object, 581 distinct): ['25 red roses', 'French roses', 'A delicate bouquet of bush roses and eustoma combined with pistachios', 'Daisies', 'Bouquet with blue orchid dendrobium combined with pistachio greens', 'Red roses 51 pieces', 'Peony-shaped roses with carnation', 'Peonies are a compliment', 'Bouquet of cosmic orchids dendrobium', 'Bouquet of sunflowers']
rating_by_comments (float64, 37 distinct): ['4.85', '4.89', '4.87', '4.83', '4.9', '4.84', '4.88', '4.92', '4.79', '4.82']
price_rub (int64, 348 distinct): ['4990', '3990', '3500', '4500', '3900', '3999', '3850', '3600', '3950', '3960']
price_usd (float64, 348 distinct): ['62.375', '49.875', '43.75', '56.25', '48.75', '49.9875', '48.125', '45.0', '49.375', '49.5']
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
