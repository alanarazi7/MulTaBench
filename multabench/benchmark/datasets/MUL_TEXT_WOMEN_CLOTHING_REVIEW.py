"""
Source: OpenML dataset 46659 (women_clothing_review)
        Women's e-commerce clothing reviews.

Target: Rating — 5-class (1–5 stars).
Features: Clothing ID, Age (numeric), Title, Review Text (text),
          Recommended IND, Positive Feedback Count (numeric),
          Division Name, Department Name, Class Name (categorical).
Dropped: Unnamed:_0.

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


DATASET_ID = "MUL_TEXT_WOMEN_CLOTHING_REVIEW"
SLUG_BASE = "multabench-women-clothing-review"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46659"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.MUL_TEXT_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
