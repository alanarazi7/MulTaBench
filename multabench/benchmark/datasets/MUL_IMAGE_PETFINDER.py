"""
Source: c/petfinder-adoption-prediction on Kaggle (competition)
        PetFinder pet adoption listings; predict pet age bin (multiclass).

Target: AGE_BIN (discretized age, multiclass).
Features: Type, Name, Breed, Color, Gender, Size, Health, Fee, State, Description + pet image.
Dropped: Age (used to create target), RescuerID, PetID.

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
from multabench.preprocessing.discretize import discretize_numerical


DATASET_ID = "MUL_IMAGE_PETFINDER"
SLUG_BASE = "multabench-petfinder"
KAGGLE_SOURCE = "c/petfinder-adoption-prediction"

TARGET_COL = "AGE_BIN"
IMAGE_COL = "Pet Image"

_ADOPTION_SPEED_MAP = {0: "Same Day", 1: "1-7 Days", 2: "8-30 Days", 3: "31-90 Days",
                       4: "Not adopted in 100 days"}
_COLS_TO_DROP = ["Age", "RescuerID", "PetID"]


def _map_col(df: pd.DataFrame, col: str, mapping: dict) -> pd.DataFrame:
    df[col] = df[col].apply(lambda x: mapping.get(x, ""))
    return df


def _get_pet_image(pet_id: str, images_dir: str) -> Optional[str]:
    img_name = f"{pet_id}-1.jpg"
    if exists(join(images_dir, img_name)):
        return img_name
    return None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    images_dir = join(dir_path, "train_images")
    df = pd.read_csv(join(dir_path, "train", "train.csv"))

    breed = pd.read_csv(join(dir_path, "breed_labels.csv")).set_index("BreedID")["BreedName"].to_dict()
    color = pd.read_csv(join(dir_path, "color_labels.csv")).set_index("ColorID")["ColorName"].to_dict()
    state = pd.read_csv(join(dir_path, "state_labels.csv")).set_index("StateID")["StateName"].to_dict()

    for col, mapping in [("Breed1", breed), ("Breed2", breed), ("Color1", color),
                         ("Color2", color), ("Color3", color), ("State", state)]:
        df = _map_col(df, col, mapping)

    df["AdoptionSpeed"] = df["AdoptionSpeed"].map(_ADOPTION_SPEED_MAP)
    df[IMAGE_COL] = df["PetID"].apply(lambda pid: _get_pet_image(pid, images_dir))
    df = df[df[IMAGE_COL].notna()].reset_index(drop=True)
    df[TARGET_COL] = discretize_numerical(df["Age"])
    drop = [c for c in _COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop)
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.competition_download(KAGGLE_SOURCE.replace("c/", ""))
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    images_dir = join(dir_path, "train_images")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=images_dir,
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug,
                 image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    _args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID}")
    curate(output_dir=_args.output_dir, slug=_args.slug)
