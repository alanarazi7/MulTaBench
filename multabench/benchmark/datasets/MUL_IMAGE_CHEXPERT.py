"""
Dataset Name: MUL_IMAGE_CHEXPERT
====
Examples: 46437
====
URL: https://www.kaggle.com/datasets/ashery/chexpert
====
Target Variable: Cardiomegaly (float64, 3 distinct): ['1.0', '0.0', '-1.0']
====
Features:

Sex (object, 2 distinct): ['Male', 'Female']
Age (int64, 74 distinct): ['90', '61', '60', '63', '66', '64', '62', '56', '58', '68']
Frontal/Lateral (object, 2 distinct): ['Frontal', 'Lateral']
AP/PA (object, 3 distinct, 17.8% missing): ['AP', 'PA', 'LL']
No Finding (float64, 2 distinct, 92.3% missing): ['1.0', '0.0']
Enlarged Cardiomediastinum (float64, 3 distinct, 87.6% missing): ['1.0', '0.0', '-1.0']
Lung Opacity (float64, 3 distinct, 51.7% missing): ['1.0', '0.0', '-1.0']
Lung Lesion (float64, 3 distinct, 95.0% missing): ['1.0', '0.0', '-1.0']
Edema (float64, 3 distinct, 44.3% missing): ['1.0', '0.0', '-1.0']
Consolidation (float64, 3 distinct, 63.8% missing): ['0.0', '-1.0', '1.0']
Pneumonia (float64, 3 distinct, 89.3% missing): ['-1.0', '1.0', '0.0']
Atelectasis (float64, 3 distinct, 70.8% missing): ['-1.0', '1.0', '0.0']
Pneumothorax (float64, 3 distinct, 71.3% missing): ['0.0', '1.0', '-1.0']
Pleural Effusion (float64, 3 distinct, 34.1% missing): ['1.0', '0.0', '-1.0']
Pleural Other (float64, 3 distinct, 96.5% missing): ['1.0', '-1.0', '0.0']
Fracture (float64, 3 distinct, 94.7% missing): ['1.0', '0.0', '-1.0']
Support Devices (float64, 3 distinct, 46.2% missing): ['1.0', '0.0', '-1.0']
X-Ray Image (object, 46437 distinct): ['images/train_patient00002_study2_view1_frontal.jpg', 'images/train_patient00005_study1_view1_frontal.jpg', 'images/train_patient00005_study1_view2_lateral.jpg', 'images/train_patient00007_study1_view1_frontal.jpg', 'images/train_patient00009_study1_view1_frontal.jpg', 'images/train_patient00009_study1_view2_lateral.jpg', 'images/train_patient00011_study12_view1_frontal.jpg', 'images/train_patient00012_study2_view1_frontal.jpg', 'images/train_patient00012_study2_view2_lateral.jpg', 'images/train_patient00017_study1_view1_frontal.jpg']
"""

import os
from os.path import join, exists
from typing import Optional

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "MUL_IMAGE_CHEXPERT"
SLUG_BASE = "multabench-chexpert"
KAGGLE_SOURCE = "ashery/chexpert"

TARGET_COL = "Cardiomegaly"
IMAGE_COL = "X-Ray Image"
IMAGE_SUBFOLDER = ""

_PREFIX = "CheXpert-v1.0-small/"


def _get_img_path(path: str, dir_path: str) -> Optional[str]:
    if not path.startswith(_PREFIX):
        return None
    rel = path.replace(_PREFIX, "")
    if not exists(join(dir_path, rel)):
        return None
    return rel


def _load_and_process(dir_path: str) -> pd.DataFrame:
    train = pd.read_csv(join(dir_path, "train.csv"))
    valid = pd.read_csv(join(dir_path, "valid.csv"))
    df = pd.concat([train, valid], ignore_index=True)
    df[IMAGE_COL] = df["Path"].apply(lambda p: _get_img_path(p, dir_path))
    df = df.drop(columns=["Path"])
    df = df[df[IMAGE_COL].notna()].reset_index(drop=True)
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=dir_path,
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug,
                 image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    _args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID}")
    curate(output_dir=_args.output_dir, slug=_args.slug)
