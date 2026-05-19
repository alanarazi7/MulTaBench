"""
Dataset Name: MUL_IMAGE_PETFINDER
====
Examples: 14652
====
URL: https://www.kaggle.com/datasets/c/petfinder-adoption-prediction
====
Target Variable: AGE_BIN (object, 8 distinct): ['Quantile 12-25%', 'Quantile 0-12%', 'Quantile 25-38%', 'Quantile 62-75%', 'Quantile 88-100%', 'Quantile 50-62%', 'Quantile 75-88%', 'Quantile 38-50%']
====
Features:

Type (int64, 2 distinct): ['1', '2']
Name (object, 8910 distinct, 8.2% missing): ['Baby', 'Lucky', 'Brownie', 'No Name', 'Mimi', 'Blackie', 'Puppy', 'Max', 'Kitty', 'Oreo']
Breed1 (object, 175 distinct): ['Mixed Breed', 'Domestic Short Hair', 'Domestic Medium Hair', 'Tabby', 'Domestic Long Hair', 'Siamese', 'Persian', 'Labrador Retriever', 'Shih Tzu', 'Terrier']
Breed2 (object, 132 distinct): ['', 'Mixed Breed', 'Domestic Short Hair', 'Domestic Medium Hair', 'Tabby', 'Domestic Long Hair', 'Siamese', 'Terrier', 'Labrador Retriever', 'Persian']
Gender (int64, 3 distinct): ['2', '1', '3']
Color1 (object, 7 distinct): ['Black', 'Brown', 'Golden', 'Cream', 'Gray', 'White', 'Yellow']
Color2 (object, 7 distinct): ['', 'White', 'Brown', 'Cream', 'Gray', 'Yellow', 'Golden']
Color3 (object, 6 distinct): ['', 'White', 'Cream', 'Gray', 'Yellow', 'Golden']
MaturitySize (int64, 4 distinct): ['2', '1', '3', '4']
FurLength (int64, 3 distinct): ['1', '2', '3']
Vaccinated (int64, 3 distinct): ['2', '1', '3']
Dewormed (int64, 3 distinct): ['1', '2', '3']
Sterilized (int64, 3 distinct): ['2', '1', '3']
Health (int64, 3 distinct): ['1', '2', '3']
Quantity (int64, 19 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
Fee (int64, 74 distinct): ['0', '50', '100', '200', '150', '20', '300', '30', '250', '1']
State (object, 14 distinct): ['Selangor', 'Kuala Lumpur', 'Pulau Pinang', 'Johor', 'Perak', 'Negeri Sembilan', 'Melaka', 'Kedah', 'Pahang', 'Terengganu']
VideoAmt (int64, 9 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '7']
Description (object, 13718 distinct, 0.1% missing): ['For Adoption', 'Dog 4 Adoption', 'Cat for adoption', 'Friendly', 'Please feel free to contact us : Stuart', 'Dog for adoption', 'PLEASE RESCUE/ADOPT ME FROM KLANG POUND OR I WILL BE PUT TO DEATH BY THIS WEEK, 28/3/10. I don\'t want to die,and I will love you immensely for saving me. Help!!! Please call ----------------------------------------------------- Adoption Procedure: This dog has been caught by Majlis Perbandaran Klang, and if nobody comes forward to adopt it, it will be euthanized within a few days. Even owned dogs are also often caught, and the owners are not aware for it. Those wishing to adopt this pet from Klang Dog Pound, please follow the procedures below: 1. Drive to Pusat Kurungan Haiwan Lebuh Sultan Muhammad Kawasan Perindustrian Bandar Sultan Sulaiman Pelabuhan Klang Tel : (For Sat & Sun, opening hours are 8am - 12pm) 2. Secure a Borang Permohonan Tuntutan Anjing, Selepas Tempoh 7 hari. Complete it & ensure it is endorsed by the relevant officier & stamped with relevant chop. 3. Provide a photostated copy of your Identification Card or Passport with each application * policies & requirements stiffen day by day * Advisable to provide a copy of IC/Passport per application (Just in case) * Secure extra application if there is any inkling of additional adoption. * Don\'t expect any leniency (Even we committee members, slaves & beggars don\'t have any unless OK by big guy) 4. Please be compassionate. Put yourself in their shoes: locked inside knowing its over. THEY DO KNOW. 5. I have seen them wasted much close to D days. Don\'t tell me they didn\'t undergo heightened enxiety & despair in anticipation of the end. What\'s worse their owners never came for them. Directions to Klang Dog Pound ================================ 1) Use Kesas Highway 2) Head for North Port till you see the signboard that writes "Melbourne 14 Days", then turn Right 3) Keep Left and turn Left at traffic light 4) Stay beside flyover and turn Right at immediate traffic light 5) Drive towards Sultan Sulaiman Industrial Estate 6) Go up first set of flyover 7) Keep Left till you see Pusat Kurungan Haiwan signboard 8) Turn Left 9) Drive on till you see gravel road work beside retention pond at the right 10) Turn in and turn Right till you reach a blue-roofed pound', 'I need a new home!! Contact Furry Friends Farm if you want to adopt me.', "The lil' puppy is currently taking shelter at SPCA Seberang Perai. Those interested to adopt her may contact us via email.", 'The puppy is currently taking shelter at SPCA Seberang Perai. Please contact SPCA Seberang Perai if you are interested to adopt her as your pet.']
PhotoAmt (float64, 30 distinct): ['1.0', '2.0', '3.0', '5.0', '4.0', '6.0', '7.0', '8.0', '9.0', '10.0']
AdoptionSpeed (object, 5 distinct): ['8-30 Days', 'Not adopted in 100 days', '31-90 Days', '1-7 Days', 'Same Day']
Pet Image (object, 14652 distinct): ['images/86e1089a3-1.jpg', 'images/6296e909a-1.jpg', 'images/3422e4906-1.jpg', 'images/5842f1ff5-1.jpg', 'images/850a43f90-1.jpg', 'images/d24c30b4b-1.jpg', 'images/1caa6fcdb-1.jpg', 'images/97aa9eeac-1.jpg', 'images/c06d167ca-1.jpg', 'images/7a0942d61-1.jpg']
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
