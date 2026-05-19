"""
Dataset Name: MUL_IMAGE_CBIS_DDSM
====
Examples: 1696
====
URL: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
====
Target Variable: breast_density (int64, 4 distinct): ['2', '3', '1', '4']
====
Features:

left or right breast (object, 2 distinct): ['RIGHT', 'LEFT']
image view (object, 2 distinct): ['MLO', 'CC']
abnormality id (int64, 6 distinct): ['1', '2', '3', '4', '5', '6']
mass shape (object, 20 distinct, 0.2% missing): ['IRREGULAR', 'OVAL', 'LOBULATED', 'ROUND', 'ARCHITECTURAL_DISTORTION', 'IRREGULAR-ARCHITECTURAL_DISTORTION', 'LYMPH_NODE', 'FOCAL_ASYMMETRIC_DENSITY', 'ASYMMETRIC_BREAST_TISSUE', 'LOBULATED-IRREGULAR']
mass margins (object, 19 distinct, 3.5% missing): ['CIRCUMSCRIBED', 'ILL_DEFINED', 'SPICULATED', 'OBSCURED', 'MICROLOBULATED', 'ILL_DEFINED-SPICULATED', 'CIRCUMSCRIBED-ILL_DEFINED', 'OBSCURED-ILL_DEFINED', 'CIRCUMSCRIBED-OBSCURED', 'OBSCURED-ILL_DEFINED-SPICULATED']
assessment (int64, 6 distinct): ['4', '5', '3', '0', '2', '1']
pathology (object, 3 distinct): ['MALIGNANT', 'BENIGN', 'BENIGN_WITHOUT_CALLBACK']
subtlety (int64, 6 distinct): ['5', '4', '3', '2', '1', '0']
image file path (object, 1592 distinct): ['images/1.3.6.1.4.1.9590.100.1.2.87251504411596839017815563663575708222_1-219.jpg', 'images/1.3.6.1.4.1.9590.100.1.2.354587724213018641829708719832963731890_1-179.jpg', 'images/1.3.6.1.4.1.9590.100.1.2.383084015312187246035597241651391161847_1-124.jpg', 'images/1.3.6.1.4.1.9590.100.1.2.292605978712963936606864280561587921668_1-031.jpg', 'images/1.3.6.1.4.1.9590.100.1.2.260395375912689985505181352172038713429_1-061.jpg', 'images/1.3.6.1.4.1.9590.100.1.2.101999469712679926627011488331183444331_1-063.jpg', 'images/1.3.6.1.4.1.9590.100.1.2.10098093912338239901459163343317187533_1-285.jpg', 'images/1.3.6.1.4.1.9590.100.1.2.252718871411822036139213438773416034416_1-212.jpg', 'images/1.3.6.1.4.1.9590.100.1.2.419410657312812960613390693250292392446_1-037.jpg', 'images/1.3.6.1.4.1.9590.100.1.2.180117656012441206012066533421449364027_1-036.jpg']
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
