"""
Source: awsaf49/cbis-ddsm-breast-cancer-image-dataset on Kaggle
        CBIS-DDSM breast cancer mammography; predict breast density (4-class).

Target: breast_density (multiclass).
Features: left/right breast, image view, mass shape/margins, assessment, pathology, subtlety + image.
Dropped: patient_id, abnormality type, cropped/ROI image paths.

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata

"""

import os
from os.path import join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "MUL_IMAGE_CBIS_DDSM"
SLUG_BASE = "multabench-cbis-ddsm"
KAGGLE_SOURCE = "awsaf49/cbis-ddsm-breast-cancer-image-dataset"

TARGET_COL = "breast_density"
IMAGE_COL = "image file path"
IMAGE_SUBFOLDER = "jpeg"

_COLS_TO_DROP = ["patient_id", "abnormality type", "cropped image file path", "ROI mask file path"]


def _extract_img_path(img_id: str, jpeg_dir: str) -> str:
    assert img_id.count("/") == 3, f"Unexpected path format: {img_id}"
    _, _, relevant_id, _ = img_id.split("/")
    files = os.listdir(join(jpeg_dir, relevant_id))
    assert len(files) == 1, f"Expected 1 file in {relevant_id}, found {len(files)}"
    return join(relevant_id, files[0])


def _load_and_process(dir_path: str) -> pd.DataFrame:
    jpeg_dir = join(dir_path, IMAGE_SUBFOLDER)
    train = pd.read_csv(join(dir_path, "csv/mass_case_description_train_set.csv"))
    test = pd.read_csv(join(dir_path, "csv/mass_case_description_test_set.csv"))
    df = pd.concat([train, test], ignore_index=True)
    df[IMAGE_COL] = df[IMAGE_COL].apply(lambda p: _extract_img_path(p, jpeg_dir))
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
