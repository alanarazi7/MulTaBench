"""
Source: dhruvildave/amazon-bin-image-dataset on Kaggle
        Amazon bin images with product metadata stored in SQLite.

Target: Total Weight in Pounds (regression).
Features: product descriptions (text), Expected Quantity + image.

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata

"""

import json
import os
import sqlite3
from os.path import join
from typing import Dict, Optional

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_AMAZON_PACKAGES"
SLUG_BASE = "multabench-amazon-packages"
KAGGLE_SOURCE = "dhruvildave/amazon-bin-image-dataset"

TARGET_COL = "Total Weight in Pounds"
IMAGE_COL = "Amazon bin"
IMAGE_SUBFOLDER = "img"

_BAD_IMAGES = {"100297.jpg", "100335.jpg"}



def _get_total_weight(data: Dict) -> Optional[float]:
    items = data.get('BIN_FCSKU_DATA', {})
    total = 0.0
    for item in items.values():
        w = item.get('weight', {})
        if not w:
            return None
        weight = float(w.get('value', 0) or 0)
        quantity = item.get('quantity')
        if not quantity:
            return None
        total += weight * quantity
    return total if total > 0 else None


def _parse_product_descriptions(data: Dict) -> str:
    descriptions = []
    for item in data.values():
        name = item.get('name')
        if name:
            descriptions.append(name)
    return "; ".join(descriptions)


def _load_and_process(dir_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(join(dir_path, "metadata.sqlite"))
    df = pd.read_sql_query("SELECT * FROM metadata", conn)
    conn.close()
    df[IMAGE_COL] = df['img_id'].apply(lambda i: f"{i}.jpg")
    df = df[~df[IMAGE_COL].isin(_BAD_IMAGES)]
    df['data'] = df['data'].apply(json.loads)
    key = 'BIN_FCSKU_DATA'
    df[key] = df['data'].apply(lambda d: d.get(key, {}))
    df[TARGET_COL] = df['data'].apply(_get_total_weight)
    df['product descriptions'] = df[key].apply(_parse_product_descriptions)
    df['Expected Quantity'] = df['data'].apply(lambda d: d.get('EXPECTED_QUANTITY', None))
    df = df.drop(columns=['img_id', 'data', key])
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


