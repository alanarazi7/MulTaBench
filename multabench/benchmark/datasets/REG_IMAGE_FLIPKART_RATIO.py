"""
Dataset Name: REG_IMAGE_FLIPKART_RATIO
====
Examples: 60900
====
URL: https://www.kaggle.com/datasets/kuchhbhi/stylish-product-image-dataset
====
Target Variable: price_ratio (float64, 14349 distinct): ['0.2993', '0.4995', '0.3994', '0.2496', '0.3329', '0.2392', '0.4494', '0.3493', '0.2492', '0.1992']
====
Features:

id (str, 60900 distinct): ['images/1.png', 'images/2.png', 'images/3.png', 'images/4.png', 'images/5.png', 'images/6.png', 'images/7.png', 'images/8.png', 'images/9.png', 'images/10.png']
brand (str, 7208 distinct, 6.5% missing): ['PUMA', 'PETER ENGLAND', 'Allen Solly', 'HIGHLANDER', 'CAMPUS', 'METRONAUT', 'ADIDAS', 'Roadster', 'ASIAN', 'ARROW']
title (str, 15510 distinct): ['Men Cargos', 'Round Neck Women Blouse', 'Loafers For Men', 'Solid Men Three Fourths', 'Embroidered Semi Stitched Lehenga Choli', 'Sneakers For Men', 'Bellies For Women', 'Cotton Solid Patiala', 'Boots For Men', 'Unstitched Cotton Polyester Blend Shirt Fabric Printed']
actual_price (float64, 1360 distinct): ['999.0', '1999.0', '1499.0', '1299.0', '1599.0', '499.0', '2999.0', '799.0', '899.0', '2499.0']
"""

import os
from os.path import exists, join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_FLIPKART_RATIO"
SLUG_BASE = "multabench-full-flipkart-ratio"
KAGGLE_SOURCE = "kuchhbhi/stylish-product-image-dataset"

TARGET_COL = "price_ratio"
IMAGE_COL = "id"
IMAGE_SUBFOLDER = join("Fashion_Products_Image", "Flipkart")
# sold_price is the numerator of the target; actual_price alone does not give it away.
COLS_TO_DROP = ["img", "url", "sold_price"]


def _parse_indian_price(p) -> float:
    # '₹939', '₹8,599'
    if isinstance(p, float):
        return p
    return float(str(p).replace("₹", "").replace(",", "").strip())


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "Data - Copy.csv"))
    df[IMAGE_COL] = df[IMAGE_COL].apply(
        lambda i: f"{i}.png" if exists(join(dir_path, IMAGE_SUBFOLDER, f"{i}.png")) else None)
    df = df[df[IMAGE_COL].notna()]
    for col in ("sold_price", "actual_price"):
        df[col] = df[col].apply(_parse_indian_price)
    df[TARGET_COL] = df["sold_price"] / df["actual_price"]
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER),
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID,
                 slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID} for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
