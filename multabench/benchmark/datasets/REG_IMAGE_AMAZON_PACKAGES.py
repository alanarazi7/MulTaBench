"""
Dataset Name: REG_IMAGE_AMAZON_PACKAGES
====
Examples: 46398
====
URL: https://www.kaggle.com/datasets/dhruvildave/amazon-bin-image-dataset
====
Target Variable: Total Weight in Pounds (float64, 28817 distinct): ['1.0', '1.5', '0.6', '0.5', '1.2', '0.75', '0.9', '0.6', '0.3', '1.3']
====
Features:

Amazon bin (object, 46398 distinct): ['images/439407.jpg', 'images/337152.jpg', 'images/432703.jpg', 'images/62702.jpg', 'images/531830.jpg', 'images/13249.jpg', 'images/66619.jpg', 'images/241568.jpg', 'images/23919.jpg', 'images/178725.jpg']
product descriptions (object, 45411 distinct): ['TaoTronics TT-AH002 30W Ultrasonic Humidifier with Cool Mist, Classic Dial Knob Control, 3.5L Large Capacity, Two 360 degree Rotatable Outlets', 'Bluedio T2 Plus Turbine Wireless Bluetooth Headphones with Mic/Micro SD Card Slot/FM Radio (Blue)', "Funny Guy Mugs Shhh There's Wine In Here Ceramic Coffee Mug, White, 11-Ounce", "Ravensburger XXL Children's Globe 180 Piece Puzzleball", 'FujiFilm Instax Mini 8 with Strap and Batteries (Blue)', 'Best LED Bulb Pack of 4 by Vemotix! - 9W equivalent 75W light (3000K) / 600lm - View Angle > 270o- 30.000 Hours Extra Long Lifespan - Very Economic - 100% Satisfaction Guarantee', '3M Virtua CCS Protective Eyewear 11872-00000-20, Foam Gasket, Anti Fog Lens, Clear', 'iOttie Easy One Touch 2 Car Mount Holder for iPhone 6s Plus 6s 5s 5c Samsung Galaxy S7 Edge S6 S5 Note 5 4', 'Alice Through the Looking Glass Chromosphere Necklace', 'Bayer Advantage II for Large Cats Over 9 lbs, 6 Pack']
Expected Quantity (int64, 79 distinct): ['3', '4', '2', '5', '6', '1', '7', '8', '9', '10']
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


