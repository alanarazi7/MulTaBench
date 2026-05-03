"""
Source: OpenML dataset 46651 (product_sentiment_machine_hack)
        MachineHack product sentiment competition.

Target: Sentiment — 4-class (Cannot Say / Negative / Positive / No Sentiment).
Features: Product_Description (text), Product_Type (categorical).
Dropped: Unnamed: 0, Text_ID.

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


DATASET_ID = "MUL_TEXT_PRODUCT_SENTIMENT"
SLUG_BASE = "multabench-product-sentiment"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46651"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.MUL_TEXT_CONSUMER_PRODUCT_SENTIMENT)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
