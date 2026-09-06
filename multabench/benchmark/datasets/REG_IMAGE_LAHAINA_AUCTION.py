"""
Dataset Name: REG_IMAGE_LAHAINA_AUCTION
====
Examples: 968
====
URL: https://www.kaggle.com/datasets/quillen/artists-for-lahaina-2023
====
Target Variable: winning_bid (int64, 218 distinct): ['100', '200', '300', '500', '400', '250', '150', '325', '350', '275']
====
Features:

artist (str, 804 distinct): ['Clark Mitchell', 'Mary Spain', 'Morgan Samuel Price', 'Valeh Levy', 'Amalia Fisch', 'Debra Huse', 'Heath Quiel', 'John Burton', 'Kathy Ramirez', 'Lisa Mcknett']
title (str, 941 distinct, 0.2% missing): ['Lahaina Harbor', 'West Maui Mountains', 'After the Storm', 'Hope', 'Maui Sunset', 'Maui Morning', 'Iao Valley', 'Tucked In', 'Pacific Coast', 'A Happy Place']
medium (str, 6 distinct): ['Oil', 'Acrylic', 'Watercolor', 'Pastel', 'Other', 'Gouache']
dim1 (float64, 44 distinct, 2.4% missing): ['8.0', '12.0', '11.0', '9.0', '16.0', '6.0', '10.0', '14.0', '20.0', '18.0']
dim2 (float64, 48 distinct, 2.4% missing): ['12.0', '10.0', '14.0', '16.0', '20.0', '8.0', '24.0', '6.0', '9.0', '11.0']
value (float64, 171 distinct, 0.2% missing): ['400.0', '500.0', '300.0', '1200.0', '450.0', '350.0', '250.0', '800.0', '200.0', '900.0']
buy_now (bool, 2 distinct): ['0', '1']
Gallery (str, 3 distinct): ['Gallery Ekolu', 'Gallery Elua', 'Gallery Ekahi']
file_name (str, 968 distinct): ['images/Hamoa Beach, Maui - Aaron Schuerr.jpg', 'images/Honokahua Bay - Aaron Schuerr.jpg', 'images/Budding Elegance - Aditi Sharma.jpg', 'images/West Maui Mountains as viewed from Lahaina - Agata Zbik.jpg', 'images/Four Shades of Yellow - Aimee Erickson.jpg', 'images/Happy Hour - Aimee Erickson.jpg', 'images/After the Storm - Alan Wayne.jpg', 'images/Breath of Spring - Alan Wolton.jpg', 'images/Miss Kathy - Alana Knuff.jpg', 'images/Grazing - Alicia van Thiel.jpg']
"""

import os
from os.path import exists, join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_LAHAINA_AUCTION"
SLUG_BASE = "multabench-full-lahaina-auction"
KAGGLE_SOURCE = "quillen/artists-for-lahaina-2023"

TARGET_COL = "winning_bid"
IMAGE_COL = "file_name"
MAIN_DIR = "data"
IMAGE_SUBFOLDER = join(MAIN_DIR, "imgs")
# img_url points at the auction site; the images ship with the dataset.
COLS_TO_DROP = ["img_url"]


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, MAIN_DIR, "clean_data.csv"))
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = df[df[IMAGE_COL].notna()]
    df = df[df[IMAGE_COL].apply(lambda i: exists(join(dir_path, IMAGE_SUBFOLDER, str(i))))]
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
