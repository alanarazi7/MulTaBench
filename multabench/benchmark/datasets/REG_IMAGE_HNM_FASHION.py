"""
Source: odins0n/handm-dataset-128x128 on Kaggle
        H&M fashion articles with images; target is average consumer age at purchase.

Target: AvgConsumerAge (regression — mean buyer age per article).
Features: prod_name, product_type_name, department_name, detail_desc (text) + image.
Dropped: numeric ID columns, redundant codes.

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata

"""

import os
from os.path import join, exists
from typing import Optional

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_HNM_FASHION"
SLUG_BASE = "multabench-hnm-fashion"
KAGGLE_SOURCE = "odins0n/handm-dataset-128x128"

TARGET_COL = "AvgConsumerAge"
IMAGE_COL = "ProductPic"
IMAGE_SUBFOLDER = "images_128_128"

_COLS_TO_DROP = ["product_code", "product_type_no", "graphical_appearance_no", "colour_group_code",
                 "perceived_colour_value_id", "perceived_colour_master_id", "department_no", "index_code",
                 "index_group_no", "section_no", "garment_group_no"]



def _transform_article_id(article_id: int) -> str:
    return "0" + str(article_id)


def _collect_image(article_id: int, dir_path: str) -> Optional[str]:
    aid = _transform_article_id(article_id)
    prefix = aid[:3]
    path = join(prefix, aid + ".jpg")
    if exists(join(dir_path, IMAGE_SUBFOLDER, path)):
        return path
    return None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "articles.csv"))
    df[IMAGE_COL] = df['article_id'].apply(lambda a: _collect_image(a, dir_path))
    df = df[df[IMAGE_COL].notna()].reset_index(drop=True)

    # Compute average consumer age per article from transactions
    transactions = pd.read_csv(join(dir_path, "transactions_train.csv"))
    transactions['article_id'] = transactions['article_id'].apply(_transform_article_id)
    df['article_id'] = df['article_id'].apply(_transform_article_id)

    customers = pd.read_csv(join(dir_path, "customers.csv"))
    tx_age = transactions[['customer_id', 'article_id']].merge(
        customers[['customer_id', 'age']], on='customer_id')
    article_age = tx_age.groupby('article_id')['age'].mean().reset_index()
    df = df.merge(article_age, on='article_id', how='inner')
    df = df.rename(columns={'age': TARGET_COL})
    df = df.drop(columns=['article_id'])

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
