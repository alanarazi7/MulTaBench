"""
Dataset Name: MUL_IMAGE_REDDIT_MEMES
====
Examples: 4821
====
URL: https://www.kaggle.com/datasets/musadiqpashak/reddit-memes-dataset
====
Target Variable: Category (str, 7 distinct): ['ProgrammerHumor', 'Animemes', 'Funnymemes', 'okbuddyretard', 'dankmemes', 'shitposting', 'engineeringmemes']
====
Features:

Total Karma (int64, 3215 distinct): ['12974036', '124293', '579249', '909765', '23817', '45479', '66484', '144587', '790000', '108779']
Comment Karma (int64, 2456 distinct): ['695', '10820', '366448', '4594', '5042', '19481', '117675', '1735', '30841', '514']
Cake Day (str, 1767 distinct): ['07-03-2025', '29-08-2020', '06-12-2016', '28-06-2024', '15-09-2024', '06-12-2023', '14-01-2025', '14-03-2025', '19-03-2025', '16-06-2016']
Upvote Ratio (float64, 89 distinct): ['0.99', '0.98', '0.97', '0.96', '0.95', '0.94', '1.0', '0.93', '0.91', '0.92']
Number of Comments (int64, 280 distinct): ['1', '2', '3', '4', '0', '5', '6', '8', '7', '9']
Time of Day (str, 4 distinct): ['Night', 'Afternoon', 'Morning', 'Evening']
File Name (str, 4821 distinct): ['images/1_SachiMod_Animemes.png', 'images/2_GroovyChirpy_Animemes.jpeg', 'images/3_kf1035_Animemes.jpeg', 'images/4_KaySanTheBrightStar_Animemes.jpeg', 'images/5_Jackabing_Animemes.jpeg', 'images/6_IllustriousFox5135_Animemes.jpeg', 'images/7_iWILLpissINuranus_Animemes.png', 'images/8_Rubikx107_Animemes.png', 'images/9_Prince0x_Animemes.jpeg', 'images/10_MakotoKurume_Animemes.jpeg']
"""

import os
from os.path import exists, join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "MUL_IMAGE_REDDIT_MEMES"
SLUG_BASE = "multabench-full-reddit-memes"
KAGGLE_SOURCE = "musadiqpashak/reddit-memes-dataset"

TARGET_COL = "Category"
IMAGE_COL = "File Name"
_MAIN_DIR = "reddit_memes_dataset"
IMAGE_SUBFOLDER = join(_MAIN_DIR, "memes")
# Upvotes fixes the ratio the subreddit is ranked by; the text fields restate the meme itself.
COLS_TO_DROP = ["Post URL", "Created Time", "Upvotes", "Extracted Text", "Title", "Author"]


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, _MAIN_DIR, "data.csv"))
    df = df[df[IMAGE_COL].apply(lambda i: exists(join(dir_path, IMAGE_SUBFOLDER, str(i))))]
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
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
