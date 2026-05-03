"""
Curation script for REG_IMAGE_SOCIAL_LETTERBOXD_MOVIES_RATING.

Source: gsimonx37/letterboxd on Kaggle
        Letterboxd movie listings with poster images and structured metadata.
        Filtered to movies released from 2021 onward with available posters.

Target: rating — average Letterboxd user rating (regression, ~0–5 scale).
Features: year, minute (runtime), tagline, primary/spoken language, genre binary
          columns + image (movie poster).
Dropped: id (identifier), description (long free-text, kept separate).

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata

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
