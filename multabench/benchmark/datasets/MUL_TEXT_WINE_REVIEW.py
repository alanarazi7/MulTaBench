"""
Source: OpenML dataset 46653 (wine_reviews)
        Wine reviews scraped from WineEnthusiast.

Target: variety — 30-class wine grape variety.
Features: country, description (text), points, price (numeric),
          province (categorical).

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


DATASET_ID = "MUL_TEXT_WINE_REVIEW"
SLUG_BASE = "multabench-wine-review"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46653"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.MUL_TEXT_FOOD_WINE_REVIEW)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
