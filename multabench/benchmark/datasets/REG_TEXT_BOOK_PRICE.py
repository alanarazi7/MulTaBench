"""
Source: OpenML dataset 46663 (bookprice_prediction)
        Book listings with price, ratings, and descriptions.

Target: Price in USD — book price in USD; renamed from raw "Price".
Features: Title, Author (text), Edition, Reviews, Ratings (numeric),
          Synopsis (text), Genre, BookCategory (categorical).

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


DATASET_ID = "REG_TEXT_BOOK_PRICE"
SLUG_BASE = "multabench-book-price"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46663"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.REG_TEXT_CONSUMER_BOOK_PRICE_PREDICTION)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
