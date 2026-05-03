"""
Source: OpenML dataset 46660 (mercari_price_suggestion100K)
        100K sample from the Mercari price suggestion challenge.

Target: log_price — log-transformed listing price (numeric).
Features: name (text), item_condition_id (numeric), category_name,
          brand_name (categorical), shipping (binary), item_description (text),
          cat1/cat2/cat3 hierarchical category split (categorical).
Dropped: train_id, price (leakage — log_price is derived from it).

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


DATASET_ID = "REG_TEXT_MERCARI_MARKETPLACE"
SLUG_BASE = "multabench-mercari-marketplace"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46660"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.REG_TEXT_CONSUMER_MERCARI_ONLINE_MARKETPLACE)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
