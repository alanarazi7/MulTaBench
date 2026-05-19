"""
Dataset Name: REG_IMAGE_LETTERBOXD_MOVIES
====
Examples: 12564
====
URL: https://www.kaggle.com/datasets/gsimonx37/letterboxd
====
Target Variable: rating (float64, 314 distinct): ['3.4', '3.41', '3.26', '3.34', '3.46', '3.43', '3.36', '3.47', '3.52', '3.49']
====
Features:

name (object, 12336 distinct): ['Last Summer', 'The Teacher', 'The Line', 'Prey', 'Hero', 'Human Resources', 'Mercy', 'The Visitor', 'First Love', 'Noise']
date (float64, 4 distinct): ['2022.0', '2023.0', '2021.0', '2024.0']
tagline (object, 4460 distinct, 64.3% missing): ['Face your demons.', 'The hunt is on.', 'Back for seconds.', 'Reap what you sow.', 'No one just disappears.', "It's only a matter of time.", 'Defy the odds', 'Find your voice.', 'Welcome to the family.', 'Time is running out.']
minute (float64, 484 distinct, 1.9% missing): ['90.0', '84.0', '100.0', '95.0', '93.0', '96.0', '97.0', '91.0', '98.0', '85.0']
themes (object, 1757 distinct): ['', 'Epic heroes; Bollywood emotional dramas', 'Song and dance; Bollywood emotional dramas', 'Moving relationship stories; Powerful stories of heartbreak and suffering', 'Epic heroes; Superheroes in action-packed battles with villains', 'Moving relationship stories; Bollywood emotional dramas', 'Thrillers and murder mysteries; Suspenseful crime thrillers; Intriguing and suspenseful murder mysteries', 'Relationship comedy; Laugh-out-loud relationship entanglements', 'Moving relationship stories; Touching and sentimental family stories', 'Intense violence and sexual transgression; Twisted dark psychological thriller']
Primary language (object, 86 distinct): ['', 'English', 'French', 'Spanish', 'German', 'Hindi', 'Korean', 'Italian', 'Chinese', 'Japanese']
Spoken language (object, 99 distinct): ['', 'Spanish', 'English', 'French', 'Portuguese', 'Italian', 'German', 'Russian', 'Japanese', 'Swedish']
genre_Action (int64, 2 distinct): ['0', '1']
genre_Adventure (int64, 2 distinct): ['0', '1']
genre_Animation (int64, 2 distinct): ['0', '1']
genre_Comedy (int64, 2 distinct): ['0', '1']
genre_Crime (int64, 2 distinct): ['0', '1']
genre_Documentary (int64, 2 distinct): ['0', '1']
genre_Drama (int64, 2 distinct): ['0', '1']
genre_Family (int64, 2 distinct): ['0', '1']
genre_Fantasy (int64, 2 distinct): ['0', '1']
genre_History (int64, 2 distinct): ['0', '1']
genre_Horror (int64, 2 distinct): ['0', '1']
genre_Music (int64, 2 distinct): ['0', '1']
genre_Mystery (int64, 2 distinct): ['0', '1']
genre_Romance (int64, 2 distinct): ['0', '1']
genre_Science Fiction (int64, 2 distinct): ['0', '1']
genre_TV Movie (int64, 2 distinct): ['0', '1']
genre_Thriller (int64, 2 distinct): ['0', '1']
genre_War (int64, 2 distinct): ['0', '1']
genre_Western (int64, 2 distinct): ['0', '1']
poster (object, 12564 distinct): ['images/1000001.jpg', 'images/1000003.jpg', 'images/1000006.jpg', 'images/1000009.jpg', 'images/1000013.jpg', 'images/1000019.jpg', 'images/1000020.jpg', 'images/1000021.jpg', 'images/1000027.jpg', 'images/1000028.jpg']
"""

import os
from collections import defaultdict
from os.path import exists, join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_LETTERBOXD_MOVIES"
SLUG_BASE = "multabench-letterboxd-movies"
KAGGLE_SOURCE = "gsimonx37/letterboxd"

TARGET_COL = "rating"
IMAGE_COL = "poster"
IMAGE_SUBFOLDER = "posters"

MIN_YEAR = 2021
COLS_TO_DROP = ["id", "description"]



def _add_themes(df: pd.DataFrame, dir_path: str) -> pd.DataFrame:
    themes = pd.read_csv(join(dir_path, "themes.csv"))
    id2themes = defaultdict(list)
    for _, row in themes.iterrows():
        id2themes[row["id"]].append(row["theme"])
    df["themes"] = df["id"].map(lambda i: "; ".join(id2themes.get(i, [])))
    return df


def _add_languages(df: pd.DataFrame, dir_path: str) -> pd.DataFrame:
    languages = pd.read_csv(join(dir_path, "languages.csv"))
    for col in ["Primary language", "Spoken language"]:
        subset = languages[languages["type"] == col]
        id2lang = dict(zip(subset["id"], subset["language"]))
        df[col] = df["id"].map(lambda i: id2lang.get(i, ""))
    return df


def _add_genres(df: pd.DataFrame, dir_path: str) -> pd.DataFrame:
    genres = pd.read_csv(join(dir_path, "genres.csv"))
    for genre in sorted(genres["genre"].unique()):
        id_set = set(genres[genres["genre"] == genre]["id"])
        df[f"genre_{genre}"] = df["id"].apply(lambda i: 1 if i in id_set else 0)
    return df


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "movies.csv"))
    df = df[df["date"] >= MIN_YEAR]
    df = df[df[TARGET_COL].notna()]
    df = _add_themes(df, dir_path)
    df = _add_languages(df, dir_path)
    df = _add_genres(df, dir_path)
    df[IMAGE_COL] = df["id"].apply(lambda i: f"{i}.jpg" if exists(join(dir_path, IMAGE_SUBFOLDER, f"{i}.jpg")) else None)
    df = df[df[IMAGE_COL].notna()].reset_index(drop=True)
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER), dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug,
                 image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
