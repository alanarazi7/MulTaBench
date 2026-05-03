"""
Source: URL — http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/
        Rotten Tomatoes movie listings with ratings, cast, and descriptions.

Target: RatingValue — audience/critic score (numeric, 0–10 scale).
Features: Name, Director, Creator, Actors, Cast, Description (text),
          Year, Release Date, Language, Country, Genre,
          Filming Locations (categorical), Duration (numeric; parsed
          from "XX min" string), RatingCount, ReviewCount (numeric).
Dropped: Id (identifier).

Produces:
    <output_dir>/
        data.csv          features + target
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import UrlDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_ROTTEN_TOMATOES"
SLUG_BASE = "multabench-rotten-tomatoes"
KAGGLE_SOURCE = "http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(UrlDatasetID.REG_TEXT_SOCIAL_MOVIES_ROTTEN_TOMATOES)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
