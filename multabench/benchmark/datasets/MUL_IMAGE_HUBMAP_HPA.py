"""
Source: miquel0/hubmaphba-tiled-dataset-512x512 on Kaggle
        HuBMAP + HPA tiled biopsy images; predict donor age category (multiclass).

Target: age (discretized into quantile bins, multiclass).
Features: organ, rle, data_source stats + biopsy image.
Dropped: id, idx, pixel_size, tissue_thickness, data_source, img dimensions.

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
from multabench.preprocessing.discretize import discretize_numerical


DATASET_ID = "MUL_IMAGE_HUBMAP_HPA"
SLUG_BASE = "multabench-hubmap-hpa"
KAGGLE_SOURCE = "miquel0/hubmaphba-tiled-dataset-512x512"

TARGET_COL = "age"
IMAGE_COL = "biopsy"
IMAGE_SUBFOLDER = "train_tiles_images"

_COLS_TO_DROP = ["id", "idx", "pixel_size", "tissue_thickness", "data_source", "img_height", "img_width"]


def _collect_img(row: pd.Series, dir_path: str) -> Optional[str]:
    idx = str(row["idx"]).zfill(3)
    img_name = f"{row['id']}_{idx}.png"
    if not exists(join(dir_path, IMAGE_SUBFOLDER, img_name)):
        return None
    return img_name


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "train_tiles.csv"))
    df[IMAGE_COL] = df.apply(lambda row: _collect_img(row, dir_path), axis=1)
    df = df[df[IMAGE_COL].notna()].reset_index(drop=True)
    df[TARGET_COL] = discretize_numerical(df[TARGET_COL])
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


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    _args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID}")
    curate(output_dir=_args.output_dir, slug=_args.slug)
