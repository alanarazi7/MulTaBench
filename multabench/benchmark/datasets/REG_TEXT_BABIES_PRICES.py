"""
Source: URL — http://pages.cs.wisc.edu/~anhai/data/784_data/baby_products/
        Babies R Us product catalogue with prices.

Target: price — product price (numeric, USD).
Features: title (text), is_discounted (binary), category, company_struct,
          company_free (categorical), weight, length, width, height
          (text/numeric), fabrics, colors, materials (categorical).
Dropped: int_id, ext_id, SKU, brand.

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


DATASET_ID = "REG_TEXT_BABIES_PRICES"
SLUG_BASE = "multabench-babies-prices"
KAGGLE_SOURCE = "http://pages.cs.wisc.edu/~anhai/data/784_data/baby_products/csv_files/babies_r_us.csv"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(UrlDatasetID.REG_TEXT_CONSUMER_BABIES_R_US_PRICES)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
