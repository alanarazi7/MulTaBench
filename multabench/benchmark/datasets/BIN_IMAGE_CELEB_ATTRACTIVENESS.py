"""
Dataset Name: BIN_IMAGE_CELEB_ATTRACTIVENESS
====
Examples: 99999
====
Target Variable: Attractive (int64, 2 distinct): ['1', '-1']
====
Features:

image_id (object, 99999 distinct): ['images/093242.jpg', 'images/121414.jpg', 'images/010625.jpg', 'images/082524.jpg', 'images/073193.jpg', 'images/176200.jpg', 'images/041602.jpg', 'images/065466.jpg', 'images/019028.jpg', 'images/164498.jpg']
5_o_Clock_Shadow (int64, 2 distinct): ['-1', '1']
Arched_Eyebrows (int64, 2 distinct): ['-1', '1']
Bags_Under_Eyes (int64, 2 distinct): ['-1', '1']
Bald (int64, 2 distinct): ['-1', '1']
Bangs (int64, 2 distinct): ['-1', '1']
Big_Lips (int64, 2 distinct): ['-1', '1']
Big_Nose (int64, 2 distinct): ['-1', '1']
Black_Hair (int64, 2 distinct): ['-1', '1']
Blond_Hair (int64, 2 distinct): ['-1', '1']
Blurry (int64, 2 distinct): ['-1', '1']
Brown_Hair (int64, 2 distinct): ['-1', '1']
Bushy_Eyebrows (int64, 2 distinct): ['-1', '1']
Chubby (int64, 2 distinct): ['-1', '1']
Double_Chin (int64, 2 distinct): ['-1', '1']
Eyeglasses (int64, 2 distinct): ['-1', '1']
Goatee (int64, 2 distinct): ['-1', '1']
Gray_Hair (int64, 2 distinct): ['-1', '1']
Heavy_Makeup (int64, 2 distinct): ['-1', '1']
High_Cheekbones (int64, 2 distinct): ['-1', '1']
Male (int64, 2 distinct): ['-1', '1']
Mouth_Slightly_Open (int64, 2 distinct): ['-1', '1']
Mustache (int64, 2 distinct): ['-1', '1']
Narrow_Eyes (int64, 2 distinct): ['-1', '1']
No_Beard (int64, 2 distinct): ['1', '-1']
Oval_Face (int64, 2 distinct): ['-1', '1']
Pale_Skin (int64, 2 distinct): ['-1', '1']
Pointy_Nose (int64, 2 distinct): ['-1', '1']
Receding_Hairline (int64, 2 distinct): ['-1', '1']
Rosy_Cheeks (int64, 2 distinct): ['-1', '1']
Sideburns (int64, 2 distinct): ['-1', '1']
Smiling (int64, 2 distinct): ['-1', '1']
Straight_Hair (int64, 2 distinct): ['-1', '1']
Wavy_Hair (int64, 2 distinct): ['-1', '1']
Wearing_Earrings (int64, 2 distinct): ['-1', '1']
Wearing_Hat (int64, 2 distinct): ['-1', '1']
Wearing_Lipstick (int64, 2 distinct): ['-1', '1']
Wearing_Necklace (int64, 2 distinct): ['-1', '1']
Wearing_Necktie (int64, 2 distinct): ['-1', '1']
Young (int64, 2 distinct): ['1', '-1']
"""

import os
from os.path import join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "BIN_IMAGE_CELEB_ATTRACTIVENESS"
SLUG_BASE = "multabench-celeb-attractiveness"
KAGGLE_SOURCE = "jessicali9530/celeba-dataset"

TARGET_COL = "Attractive"
IMAGE_COL = "image_id"
IMAGE_SUBFOLDER = "img_align_celeba/img_align_celeba"
SAMPLE_N = 100_000
RANDOM_STATE = 42



def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "list_attr_celeba.csv"))
    df = df.sample(n=SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER), dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID), kaggle_source=KAGGLE_SOURCE)


