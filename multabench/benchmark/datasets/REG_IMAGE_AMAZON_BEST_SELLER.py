"""
Dataset Name: REG_IMAGE_AMAZON_BEST_SELLER
====
Examples: 3488
====
URL: https://www.kaggle.com/datasets/amankumar20d/amazon-best-seller-all-departments-us
====
Target Variable: price (float64, 1151 distinct): ['2.397', '2.1961', '3.044', '2.772', '2.3016', '3.4337', '3.7133', '2.0782', '1.7901', '3.9318']
====
Features:

num_ratings (float64, 2656 distinct, 9.7% missing): ['777049.0', '1.0', '6.0', '3.0', '12.0', '14.0', '2.0', '10.0', '4.0', '53489.0']
photo_url (object, 3446 distinct): ['images/https___images_na.ssl_images_amazon.com_images_I_81bpKKv68_L._AC_UL300_SR300_200_.jpg', 'images/https___images_na.ssl_images_amazon.com_images_I_913C_MR3S5L._AC_UL300_SR300_200_.jpg', 'images/https___images_na.ssl_images_amazon.com_images_I_61Cme7jI2eL._AC_UL300_SR300_200_.jpg', 'images/https___images_na.ssl_images_amazon.com_images_I_818i3AJdNdL._AC_UL300_SR300_200_.jpg', 'images/https___images_na.ssl_images_amazon.com_images_I_61bJZx1v8GL._AC_UL300_SR300_200_.jpg', 'images/https___images_na.ssl_images_amazon.com_images_I_51D5YPHy0hL._AC_UL300_SR300_200_.jpg', 'images/https___images_na.ssl_images_amazon.com_images_I_71n2yIfcpRL._AC_UL300_SR300_200_.jpg', 'images/https___images_na.ssl_images_amazon.com_images_I_619mM1ncz4L._AC_UL300_SR300_200_.jpg', 'images/https___images_na.ssl_images_amazon.com_images_I_815FWesbK3L._AC_UL300_SR300_200_.jpg', 'images/https___images_na.ssl_images_amazon.com_images_I_91YrLTBnMcL._AC_UL300_SR300_200_.jpg']
rank (int64, 100 distinct): ['4', '3', '10', '7', '32', '34', '36', '77', '81', '54']
star_rating (float64, 35 distinct, 2.1% missing): ['4.7', '4.6', '4.8', '4.5', '4.4', '4.3', '4.2', '4.1', '4.9', '4.0']
page (int64, 2 distinct): ['2', '1']
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
