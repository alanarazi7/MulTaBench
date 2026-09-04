"""
Source: https://www.openml.org/search?type=data&id=46667

800 rows. IMDB movies; predict whether a film is a drama.

Target: Genre — binary (Genre_is_Drama, 1 = drama).
Features: Plot description and title text, director, star cast, runtime, rating, votes,
          revenue and Metascore.

Produces:
    <output_dir>/
        data.csv          features + target
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import OpenMLDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "BIN_TEXT_IMDB_GENRE"
SLUG_BASE = "multabench-full-imdb-genre"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46667"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.BIN_TEXT_SOCIAL_IMDB_GENRE_PREDICTION)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate BIN_TEXT_IMDB_GENRE for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
