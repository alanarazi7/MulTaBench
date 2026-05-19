"""
Dataset Name: MUL_IMAGE_GLAUCOMA_SMDG
====
Examples: 12449
====
URL: https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset
====
Target Variable: types (int64, 3 distinct): ['0', '1', '-1']
====
Features:

fundus (object, 12449 distinct): ['images/full-fundus_full-fundus_OIA-ODIR-TEST-OFFLINE-1.png', 'images/full-fundus_full-fundus_OIA-ODIR-TEST-OFFLINE-2.png', 'images/full-fundus_full-fundus_OIA-ODIR-TEST-OFFLINE-4.png', 'images/full-fundus_full-fundus_OIA-ODIR-TEST-OFFLINE-5.png', 'images/full-fundus_full-fundus_OIA-ODIR-TEST-OFFLINE-6.png', 'images/full-fundus_full-fundus_OIA-ODIR-TEST-OFFLINE-7.png', 'images/full-fundus_full-fundus_OIA-ODIR-TEST-OFFLINE-8.png', 'images/full-fundus_full-fundus_OIA-ODIR-TEST-OFFLINE-9.png', 'images/full-fundus_full-fundus_OIA-ODIR-TEST-OFFLINE-10.png', 'images/full-fundus_full-fundus_OIA-ODIR-TEST-OFFLINE-11.png']
sex (object, 2 distinct, 63.7% missing): ['M', 'F']
age (float64, 77 distinct, 59.8% missing): ['56.0', '60.0', '55.0', '62.0', '54.0', '65.0', '64.0', '59.0', '63.0', '66.0']
eye (object, 2 distinct, 53.9% missing): ['OS', 'OD']
sbp (float64, 20 distinct, 99.8% missing): ['156.0', '153.0', '177.0', '167.0', '176.0', '127.0', '136.0', '162.0', '130.0', '123.0']
dbp (float64, 15 distinct, 99.8% missing): ['80.0', '79.0', '76.0', '81.0', '70.0', '95.0', '47.0', '64.0', '86.0', '90.0']
hr (float64, 16 distinct, 99.8% missing): ['68.0', '51.0', '66.0', '67.0', '63.0', '75.0', '52.0', '62.0', '84.0', '64.0']
iop (float64, 11 distinct, 99.8% missing): ['15.0', '13.0', '14.0', '8.0', '18.0', '12.0', '17.0', '16.0', '22.0', '19.0']
left_right (object, 2 distinct, 63.3% missing): ['left', 'right']
"""

import os
from os.path import join, exists
from typing import Optional

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "MUL_IMAGE_GLAUCOMA_SMDG"
SLUG_BASE = "multabench-glaucoma-smdg"
KAGGLE_SOURCE = "deathtrooper/multichannel-glaucoma-benchmark-dataset"

TARGET_COL = "types"
IMAGE_COL = "fundus"
IMAGE_SUBFOLDER = ""

_FULLY_MISSING = [
    'Unnamed  24', 'vcdr',
    'notchI_present', 'notchS_present', 'notchN_present', 'notchT_present',
    'expert1_grade', 'expert2_grade', 'expert3_grade', 'expert4_grade', 'expert5_grade',
    'cdr_avg', 'cdr_expert1', 'cdr_expert2', 'cdr_expert3', 'cdr_expert4',
    'refractive_dioptre_1', 'refractive_dioptre_2', 'refractive_astigmatism',
    'phakic_or_pseudophakic',
    'iop_perkins', 'iop_pneumatic', 'pachymetry', 'axial_length', 'visual_field_mean_defect',
    'type_expanded',
    'fundus_od_seg', 'fundus_oc_seg',
    'oct', 'oct_od_seg', 'oct_oc_seg',
    'gender',
]
_MOSTLY_MISSING = ['bv_seg', 'artery_seg', 'vein_seg']
_COLS_TO_DROP = ['patient_id', 'isColor', 'names', 'original_name'] + _FULLY_MISSING + _MOSTLY_MISSING


def _fix_fundus_path(fundus: str, dir_path: str) -> Optional[str]:
    if not isinstance(fundus, str) or fundus.count('/') != 2 or not fundus.startswith('/'):
        return None
    fundus = fundus[1:]
    fundus_name, file_name = fundus.split('/')
    new_path = join(fundus_name, fundus_name, file_name)
    if not exists(join(dir_path, new_path)):
        return None
    return new_path


def _get_left_right(org: str) -> Optional[str]:
    if not isinstance(org, str):
        return None
    if 'left' in org.lower():
        return 'left'
    if 'right' in org.lower():
        return 'right'
    return None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "metadata - standardized.csv"))
    df[IMAGE_COL] = df[IMAGE_COL].apply(lambda f: _fix_fundus_path(f, dir_path))
    df['left_right'] = df['original_name'].apply(_get_left_right)
    bad_cols = [c for c in df.columns if 'Unnamed' in c]
    df = df.drop(columns=bad_cols)
    df = df[df[IMAGE_COL].notna()].reset_index(drop=True)
    drop = [c for c in _COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop)
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
