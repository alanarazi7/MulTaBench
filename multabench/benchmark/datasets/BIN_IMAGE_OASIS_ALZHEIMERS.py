"""
Dataset Name: BIN_IMAGE_OASIS_ALZHEIMERS
====
Examples: 436
====
URL: https://www.kaggle.com/datasets/shreyanmohanty/oasis-alzheimers-detection-multi-class-dataset
====
Target Variable: is_demented (str, 2 distinct): ['NonDemented', 'Demented']
====
Features:

ID (str, 436 distinct): ['images/train_NonDemented_OAS1_0002_MR1_1_nii_slice_129_png.rf.cff99c6cc6fed2eeae445cf8584b4368.jpg', 'images/train_NonDemented_OAS1_0007_MR1_1_nii_slice_105_png.rf.06fd37c661334d536636bede0809db8a.jpg', 'images/train_NonDemented_OAS1_0009_MR1_1_nii_slice_107_png.rf.060d7b3436f5cd1bac63fc529afb4c8b.jpg', 'images/train_NonDemented_OAS1_0010_MR1_1_nii_slice_139_png.rf.34183aff9e59f0ac25b2323fc029ce8e.jpg', 'images/train_NonDemented_OAS1_0012_MR1_1_nii_slice_127_png.rf.3cbb026f06631bfc8aa54e042796eb4d.jpg', 'images/train_NonDemented_OAS1_0013_MR1_1_nii_slice_111_png.rf.0b6e8a63c4732b68e98cc8a97bee9312.jpg', 'images/train_NonDemented_OAS1_0014_MR1_1_nii_slice_100_png.rf.ccd1eb586f495dfc4c74927b5e197efb.jpg', 'images/train_VeryMildDemented_OAS1_0015_MR1_1-nii_slice_137_png.rf.8f3b312e684bce36c8ecd4f7a1295a09.jpg', 'images/train_VeryMildDemented_OAS1_0023_MR1_1-nii_slice_109_png.rf.3571530fee2ba24c2e33512030faeba3.jpg', 'images/train_MildDemented_OAS1_0028_MR1_1-nii_slice_113_png.rf.21bd51dca327e84999882044bbbd8ed3.jpg']
M/F (str, 2 distinct): ['F', 'M']
Age (int64, 73 distinct): ['20', '22', '21', '23', '73', '80', '25', '71', '19', '78']
Educ (float64, 5 distinct, 46.1% missing): ['2.0', '5.0', '4.0', '3.0', '1.0']
SES (float64, 5 distinct, 50.5% missing): ['2.0', '1.0', '3.0', '4.0', '5.0']
eTIV (int64, 312 distinct): ['1567', '1447', '1439', '1475', '1714', '1346', '1516', '1653', '1313', '1350']
ASF (float64, 282 distinct): ['1.19', '1.024', '1.12', '1.174', '1.165', '1.213', '1.142', '1.169', '1.073', '1.062']
Delay (float64, 14 distinct, 95.4% missing): ['2.0', '5.0', '1.0', '3.0', '40.0', '20.0', '28.0', '89.0', '64.0', '10.0']
"""

import os
from os.path import isdir, join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "BIN_IMAGE_OASIS_ALZHEIMERS"
SLUG_BASE = "multabench-full-oasis-alzheimers"
KAGGLE_SOURCE = "shreyanmohanty/oasis-alzheimers-detection-multi-class-dataset"

TARGET_COL = "is_demented"
IMAGE_COL = "ID"
_SPLITS = ("train", "test")
_ID_PREFIX_LEN = 13  # 'OAS1_0247_MR1' identifies the patient; the rest of the filename is the slice.

# CDR, MMSE and nWBV are clinical dementia gradings, i.e. the label by another name. Hand is constant.
COLS_TO_DROP = ["Hand", "CDR", "MMSE", "nWBV"]


def _patient_to_image(dir_path: str) -> dict[str, str]:
    """One representative slice per patient, keyed by the patient prefix of the filename."""
    prefix2image = {}
    for split in _SPLITS:
        for condition in sorted(os.listdir(join(dir_path, split))):
            condition_path = join(split, condition)
            if not isdir(join(dir_path, condition_path)):
                continue
            for f in sorted(os.listdir(join(dir_path, condition_path))):
                prefix2image.setdefault(f[:_ID_PREFIX_LEN], join(condition_path, f))
    return prefix2image


def _load_and_process(dir_path: str) -> pd.DataFrame:
    frames = [pd.read_csv(join(dir_path, f"oasis_{s}_patients_metadata.csv")) for s in _SPLITS]
    df = pd.concat(frames, ignore_index=True)
    assert df["ID"].is_unique, f"Expected one row per patient, got {df['ID'].duplicated().sum()} duplicates"
    df[TARGET_COL] = df["class"].apply(lambda c: "NonDemented" if c == "NonDemented" else "Demented")
    df = df.drop(columns=["class"] + COLS_TO_DROP, errors="ignore")
    df[IMAGE_COL] = df[IMAGE_COL].map(_patient_to_image(dir_path))
    df = df[df[IMAGE_COL].notna()].reset_index(drop=True)
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
