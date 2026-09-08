import json

import numpy as np
import os
from os.path import exists, join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_TOKOPEDIA_WEIGHT"
SLUG_BASE = "multabench-full-tokopedia-weight"
KAGGLE_SOURCE = "nsmlehq/tokopedia-products-2025"

TARGET_COL = "weight"
IMAGE_COL = "product_image"
_IMAGES_RAW = "images"
_VALID_SUFFIXES = (".jpg", ".jpeg", ".png")

COLS_TO_DROP = [
    "id", "category_id", "parent_product_id", "shop_id", "warehouse_id",
    "category_url", "shop_url", "shop_applink", "shop_reputation", "url", "applink", "thumbnail",
    "view_count", "talk_count", "weight_unit",
    "price_text", "price_slash_text",
    # Indonesian free text; the benchmark's encoder is English-only.
    "description",
    # The product title names the object, and naming the object gives away its mass: leaving it in
    # would let the text encoder answer the question the photograph is being asked.
    "name",
    # An Indonesian rendering of sold_count ("3 rb+" for 3 thousand).
    "sold_count_text",
]


def _first_valid_image(raw: str) -> str | None:
    """One image per row: MulTaBench artifacts carry a single image column, so the remaining
    product photos are recorded as a count rather than as extra image features."""
    for path in json.loads(raw):
        if path.endswith(_VALID_SUFFIXES):
            return path
    return None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "products.csv"))
    parsed = df[_IMAGES_RAW].apply(json.loads)
    df["image_cnt"] = parsed.apply(len)
    df[IMAGE_COL] = df[_IMAGES_RAW].apply(_first_valid_image)
    df = df.drop(columns=[_IMAGES_RAW] + COLS_TO_DROP, errors="ignore")
    df = df[df[IMAGE_COL].notna()]
    df = df[df[IMAGE_COL].apply(lambda p: exists(join(dir_path, str(p))))]
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)
    # Shipping weights run 1g to 150kg and skew 8.1; log1p brings that to 0.07.
    df[TARGET_COL] = np.log1p(df[TARGET_COL])
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=dir_path,
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID,
                 slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID} for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
