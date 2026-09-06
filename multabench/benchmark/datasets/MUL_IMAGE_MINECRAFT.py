"""
Dataset Name: MUL_IMAGE_MINECRAFT
====
Examples: 6089
====
URL: https://www.kaggle.com/datasets/sqdartemy/minecraft-screenshots-dataset-with-features
====
Target Variable: target_mob (str, 7 distinct): ['no_mob', 'other', 'skeleton', 'zombie', 'spider', 'ender', 'creeper']
====
Features:

screenshot_title (str, 5574 distinct): ['images/fighting (820).jpg', 'images/fighting (829).jpg', 'images/fighting (830).jpg', 'images/fighting (851).jpg', 'images/fighting (852).jpg', 'images/fighting (855).jpg', 'images/fighting (858).jpg', 'images/fighting (860).jpg', 'images/fighting (861).jpg', 'images/fighting (868).jpg']
activity (str, 6 distinct): ['archery', 'fighting', 'swimming', 'walking', 'mining', 'building']
hearts (int64, 21 distinct): ['20', '4', '7', '6', '15', '11', '14', '18', '5', '12']
light_lvl (str, 3 distinct): ['high', 'mid', 'low']
in_hand_item (str, 8 distinct): ['bow', 'no_item', 'sword', 'pickaxe', 'miscellaneous', 'block', 'axe', 'crossbow']
decision_activity (str, 6 distinct): ['give_resistance', 'give_strength', 'give_water_breathing', 'give_speed', 'give_haste', 'give_jump_boost']
decision_hearts (str, 5 distinct): ['no_decision_for_hearts', 'give_regeneration_3', 'give_regeneration_4', 'give_regeneration_2', 'give_regeneration_1']
decision_light (str, 3 distinct): ['no_decision_for_light', 'place_light_source', 'palce_light_source']
"""

import os
from os.path import exists, join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "MUL_IMAGE_MINECRAFT"
SLUG_BASE = "multabench-full-minecraft"
KAGGLE_SOURCE = "sqdartemy/minecraft-screenshots-dataset-with-features"

TARGET_COL = "target_mob"
IMAGE_COL = "screenshot_title"
IMAGE_SUBFOLDER = join("screenshots", "screenshots")
# decision_mob is the action taken against the target mob, i.e. the label restated.
COLS_TO_DROP = ["decision_mob"]


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "features_and_decisions.csv"))
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    missing = [i for i in df[IMAGE_COL] if not exists(join(dir_path, IMAGE_SUBFOLDER, str(i)))]
    assert not missing, f"{len(missing)} images referenced but absent, first: {missing[:3]}"
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
