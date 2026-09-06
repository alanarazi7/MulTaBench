"""
Dataset Name: REG_IMAGE_SOCAL_HOUSES
====
Examples: 15474
====
URL: https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal
====
Target Variable: price (int64, 2320 distinct): ['699000', '799000', '749000', '899000', '599000', '650000', '499000', '649000', '550000', '750000']
====
Features:

image_id (str, 15474 distinct): ['images/0.jpg', 'images/1.jpg', 'images/2.jpg', 'images/3.jpg', 'images/4.jpg', 'images/5.jpg', 'images/6.jpg', 'images/7.jpg', 'images/8.jpg', 'images/9.jpg']
street (str, 12401 distinct): ['Address not provided', '1930 W San Marcos Blvd', '65565 Acoma Avenue', '315 Verbena Drive', '650 S Rancho Santa Fe Rd', '40171 Wilson Street', '2846 Griffin Avenue', '26076 Fiesta Place', '276 N El Camino Real', '6665 Mission Gorge Rd']
citi (str, 415 distinct): ['San Diego, CA', 'Los Angeles, CA', 'Lancaster, CA', 'La Quinta, CA', 'Riverside, CA', 'Corona, CA', 'Escondido, CA', 'Fontana, CA', 'Palm Springs, CA', 'Big Bear, CA']
n_citi (int64, 415 distinct): ['320', '207', '193', '175', '310', '87', '115', '119', '266', '38']
bed (int64, 12 distinct): ['3', '4', '2', '5', '6', '1', '7', '8', '10', '9']
bath (float64, 32 distinct): ['2.0', '3.0', '2.1', '1.0', '3.1', '4.1', '4.0', '1.1', '5.1', '5.0']
sqft (int64, 3571 distinct): ['1200', '1100', '1600', '1440', '1300', '1344', '1000', '1800', '1400', '1500']
"""

import os
from os.path import exists, join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_SOCAL_HOUSES"
SLUG_BASE = "multabench-full-socal-houses"
KAGGLE_SOURCE = "ted8080/house-prices-and-images-socal"

TARGET_COL = "price"
IMAGE_COL = "image_id"
IMAGE_SUBFOLDER = join("socal2", "socal_pics")


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "socal2.csv"))
    df[IMAGE_COL] = df[IMAGE_COL].apply(lambda i: f"{i}.jpg")
    missing = [i for i in df[IMAGE_COL] if not exists(join(dir_path, IMAGE_SUBFOLDER, i))]
    assert not missing, f"{len(missing)} images referenced but absent, first: {missing[:3]}"
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
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
