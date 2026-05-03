"""
Source: deathtrooper/multichannel-glaucoma-benchmark-dataset on Kaggle
        Standardized multi-channel glaucoma benchmark; predict glaucoma type (3-class).

Target: types (multiclass: 0=normal, 1=glaucoma, -1=suspect).
Features: sex, age, eye, IOP, BP, left_right + fundus image.
Dropped: patient_id, fully/mostly missing fields, raw name columns.

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
