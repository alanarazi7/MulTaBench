"""
Source: Figshare (article 21454413), originally from the MuG paper
        https://aclanthology.org/2023.findings-emnlp.354.pdf

CS:GO skin data collected from https://www.csgodatabase.com/.
956 rows covering train/dev/test splits.

Target: Price_Category — Min Price discretized into quantile bins (multiclass).
Features: Skin Name (text), Skin Quality, Skin Category, Availability, Image Path (image).
Dropped as leakage: Min Price, Max Price.

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata

"""

import os
import tempfile
from os.path import join
from typing import Optional

import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name
from multabench.preprocessing.discretize import discretize_numerical
from multabench.utils.file_downloading import download_from_figshare, unzip_url_dataset


DATASET_ID = "MUL_IMAGE_CSGO_SKIN_PRICE"
SLUG_BASE = "multabench-csgo-skin"

FIGSHARE_ARTICLE_ID = 21454413
FIGSHARE_FILE_ID = 38077458
INNER_DIR = "CSGO-Skin-quality"
SPLITS = ["train", "dev", "test"]

TARGET_COL = "Price_Category"
IMAGE_COL = "Image Path"
KAGGLE_SOURCE = "https://figshare.com/articles/dataset/21454413"

COLS_TO_DROP = ["id", "Min Price", "Max Price"]



def _fix_price(pr: str) -> Optional[float]:
    if not isinstance(pr, str):
        return None
    pr = pr.strip().replace("$", "").replace(",", "")
    for c in pr:
        if c not in ".0123456789":
            return None
    return float(pr) if pr else None


def _download_and_unzip(dst_dir: str) -> None:
    zip_path = download_from_figshare(FIGSHARE_ARTICLE_ID, FIGSHARE_FILE_ID, "CounterStrikeGO")
    unzip_url_dataset(zip_path, dst_dir)
    for split in SPLITS:
        split_zip = join(dst_dir, INNER_DIR, f"{split}_images.zip")
        unzip_url_dataset(split_zip, join(dst_dir, INNER_DIR))
        os.remove(split_zip)


def _load_df(inner_dir: str) -> pd.DataFrame:
    dfs = [pd.read_csv(join(inner_dir, f"{split}.csv")) for split in SPLITS]
    df = pd.concat(dfs, ignore_index=True)
    df["Min Price"] = df["Min Price"].apply(_fix_price)
    df["Max Price"] = df["Max Price"].apply(_fix_price)
    df[TARGET_COL] = discretize_numerical(df["Min Price"])
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    return df



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print("Downloading and unzipping from Figshare...")
        _download_and_unzip(tmp)
        inner_dir = join(tmp, INNER_DIR)

        print("Loading and processing CSVs...")
        df = _load_df(inner_dir)
        print(f"  {len(df)} rows loaded")

        print(f"Copying images to {join(output_dir, IMAGES_DIR)} ...")
        df = copy_images(df=df, image_col=IMAGE_COL, src_dir=inner_dir, dst_dir=join(output_dir, IMAGES_DIR))

    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug,
                 image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


