import os
from os.path import exists, join

import kagglehub
import numpy as np
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name
from multabench.utils.file_downloading import download_url_image_column


DATASET_ID = "REG_IMAGE_ZEPTO_PRICE"
SLUG_BASE = "multabench-full-zepto-price"
KAGGLE_SOURCE = "simpleaditya/zepto-products-dataset"

TARGET_COL = "Price"
IMAGE_COL = "Image"
_DOWNLOAD_DIR = join(".tabstar_temp_files", "zepto_images")
# Original Price is the same figure before a markdown, so it states the target. Name spells out
# the product, which is most of the price and would leave the photograph nothing to add.
COLS_TO_DROP = ["Original Price", "Name"]


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_excel(join(dir_path, "zepto dataset.xlsx"))
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL] > 0]
    df = df[df[IMAGE_COL].astype(str).str.startswith("http")]
    # The catalogue lists one product under several pack sizes against a single photo.
    df = df.drop_duplicates(subset=IMAGE_COL, keep="first")
    df = df.drop(columns=COLS_TO_DROP, errors="ignore").reset_index(drop=True)
    df = download_url_image_column(df=df, img_folder=_DOWNLOAD_DIR, img_col=IMAGE_COL)
    df = df[df[IMAGE_COL].apply(lambda p: bool(p) and exists(join(_DOWNLOAD_DIR, str(p))))]
    df = df.reset_index(drop=True)
    df[TARGET_COL] = np.log1p(df[TARGET_COL])
    assert df[IMAGE_COL].is_unique, "One photo per product keeps the image split meaningful"
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=_DOWNLOAD_DIR,
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID,
                 slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID} for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
