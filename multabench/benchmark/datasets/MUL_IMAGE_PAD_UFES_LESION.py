"""
Dataset Name: MUL_IMAGE_PAD_UFES_LESION
====
Examples: 1641
====
URL: https://www.kaggle.com/datasets/mahdavi1202/skin-cancer
====
Target Variable: diagnostic (object, 6 distinct): ['BCC', 'ACK', 'NEV', 'SEK', 'SCC', 'MEL']
====
Features:

smoke (object, 2 distinct, 36.8% missing): ['0', '1']
drink (object, 2 distinct, 36.8% missing): ['0', '1']
background_father (object, 13 distinct, 37.3% missing): ['POMERANIA', 'GERMANY', 'ITALY', 'UNK', 'BRAZIL', 'NETHERLANDS', 'PORTUGAL', 'POLAND', 'BRASIL', 'CZECH']
background_mother (object, 11 distinct, 37.5% missing): ['POMERANIA', 'GERMANY', 'ITALY', 'UNK', 'BRAZIL', 'NETHERLANDS', 'PORTUGAL', 'POLAND', 'NORWAY', 'SPAIN']
age (int64, 84 distinct): ['73', '57', '75', '58', '71', '55', '53', '65', '70', '62']
pesticide (object, 2 distinct, 36.8% missing): ['0', '1']
gender (object, 2 distinct, 36.8% missing): ['FEMALE', 'MALE']
skin_cancer_history (object, 2 distinct, 36.8% missing): ['0', '1']
cancer_history (object, 2 distinct, 36.8% missing): ['1', '0']
has_piped_water (object, 2 distinct, 36.8% missing): ['1', '0']
has_sewage_system (object, 2 distinct, 36.8% missing): ['1', '0']
fitspatrick (float64, 6 distinct, 36.8% missing): ['2.0', '3.0', '1.0', '4.0', '5.0', '6.0']
region (object, 14 distinct): ['FACE', 'FOREARM', 'CHEST', 'BACK', 'ARM', 'NOSE', 'HAND', 'NECK', 'EAR', 'THIGH']
diameter_1 (float64, 41 distinct, 36.8% missing): ['10.0', '7.0', '6.0', '5.0', '15.0', '9.0', '8.0', '12.0', '4.0', '11.0']
diameter_2 (float64, 35 distinct, 36.8% missing): ['5.0', '10.0', '6.0', '7.0', '4.0', '8.0', '3.0', '9.0', '15.0', '12.0']
itch (object, 3 distinct): ['TRUE', 'FALSE', 'UNK']
grew (object, 3 distinct): ['FALSE', 'TRUE', 'UNK']
hurt (object, 3 distinct): ['FALSE', 'TRUE', 'UNK']
changed (object, 3 distinct): ['FALSE', 'UNK', 'TRUE']
bleed (object, 3 distinct): ['FALSE', 'TRUE', 'UNK']
elevation (object, 3 distinct): ['TRUE', 'FALSE', 'UNK']
img_id (object, 1641 distinct): ['images/imgs_part_3_imgs_part_3_PAT_1516_1765_530.png', 'images/imgs_part_1_imgs_part_1_PAT_46_881_939.png', 'images/imgs_part_3_imgs_part_3_PAT_1545_1867_547.png', 'images/imgs_part_3_imgs_part_3_PAT_1989_4061_934.png', 'images/imgs_part_2_imgs_part_2_PAT_684_1302_588.png', 'images/imgs_part_3_imgs_part_3_PAT_1549_1882_230.png', 'images/imgs_part_2_imgs_part_2_PAT_778_1471_835.png', 'images/imgs_part_1_imgs_part_1_PAT_117_179_983.png', 'images/imgs_part_3_imgs_part_3_PAT_1995_4080_695.png', 'images/imgs_part_2_imgs_part_2_PAT_705_4015_413.png']
"""

import os
from os.path import exists, join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "MUL_IMAGE_PAD_UFES_LESION"
SLUG_BASE = "multabench-full-pad-ufes-lesion"
KAGGLE_SOURCE = "mahdavi1202/skin-cancer"

TARGET_COL = "diagnostic"
IMAGE_COL = "img_id"
_IMAGE_PARTS = [join(f"imgs_part_{i}", f"imgs_part_{i}") for i in (1, 2, 3)]
# `biopsed` is a proxy: every BCC, MEL and SCC was biopsied and most benign lesions were not.
# Patient and lesion ids identify the split rather than describe the lesion.
COLS_TO_DROP = ["biopsed", "patient_id", "lesion_id"]


def _resolve_image(dir_path: str, img_id: str) -> str | None:
    for part in _IMAGE_PARTS:
        if exists(join(dir_path, part, img_id)):
            return join(part, img_id)
    return None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "metadata.csv"))
    # A lesion is photographed more than once; one row per lesion keeps it out of both splits.
    df = df.drop_duplicates(subset="lesion_id", keep="first")
    df[IMAGE_COL] = df[IMAGE_COL].apply(lambda i: _resolve_image(dir_path, str(i)))
    df = df[df[IMAGE_COL].notna()]
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    assert df[IMAGE_COL].is_unique, "Duplicate image after dedup"
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
