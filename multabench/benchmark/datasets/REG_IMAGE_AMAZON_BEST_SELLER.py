"""
Curation script for REG_IMAGE_AMAZON_BEST_SELLER_PRODUCT_RATINGS.

Source: amankumar20d/amazon-best-seller-all-departments-us on Kaggle
        ~3500 Amazon best-seller products across 40 departments with images,
        prices, ratings, and reviews.

Target: price (regression, log-transformed).
Features: Image (photo_url), num_ratings (numeric).
Dropped: asin, Unnamed: 0, url, title, department.

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata

"""

import math
import os
import re
from os.path import join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name
from multabench.utils.file_downloading import download_url_image_column


DATASET_ID = "REG_IMAGE_AMAZON_BEST_SELLER"
SLUG_BASE = "multabench-amazon-best-seller"
KAGGLE_SOURCE = "amankumar20d/amazon-best-seller-all-departments-us"

TARGET_COL = "price"
IMAGE_COL = "photo_url"
IMAGE_FOLDER = "downloaded_amazon_images"
COLS_TO_DROP = ["asin", "Unnamed: 0", "url", "title", "department"]


def _parse_price(price) -> float | None:
    if isinstance(price, float):
        return math.log1p(price)
    if not isinstance(price, str):
        return None
    try:
        return math.log1p(float(price.replace("$", "").replace(",", "")))
    except ValueError:
        return None


def _parse_num_ratings(rating) -> float | None:
    if isinstance(rating, float):
        return rating
    if not isinstance(rating, str):
        return None
    try:
        return float(rating.replace(",", ""))
    except ValueError:
        return None


def _load_and_process(dir_path: str, img_folder: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "Amazon_Best_Seller.csv"))
    df = download_url_image_column(df=df, img_folder=img_folder, img_col=IMAGE_COL)
    df[TARGET_COL] = df[TARGET_COL].apply(_parse_price)
    df["num_ratings"] = df["num_ratings"].apply(_parse_num_ratings)
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = df[df[IMAGE_COL].notna() & (df[IMAGE_COL] != "")].reset_index(drop=True)
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    return df


def _sanitize_image_names(df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
    renamed = {}
    for fname in os.listdir(images_dir):
        clean = re.sub(r"[^a-zA-Z0-9._]", "_", fname)
        if clean != fname:
            os.rename(join(images_dir, fname), join(images_dir, clean))
            renamed[fname] = clean
    if renamed:
        df = df.copy()
        df[IMAGE_COL] = df[IMAGE_COL].apply(
            lambda p: f"{IMAGES_DIR}/{renamed[os.path.basename(p)]}"
            if os.path.basename(p) in renamed else p
        )
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    img_folder = join(dir_path, IMAGE_FOLDER)
    df = _load_and_process(dir_path, img_folder)
    print(f"  {len(df)} rows loaded")
    images_dir = join(output_dir, IMAGES_DIR)
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=img_folder, dst_dir=images_dir)
    df = _sanitize_image_names(df, images_dir)
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug,
                 image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate Amazon Best Seller Product Ratings dataset")
    curate(args.output_dir, args.slug)
