"""
Source: Kaggle — himanshupoddar/zomato-bangalore-restaurants
        Zomato restaurant listings for Bangalore (~51K rows).

Target: rate — average user rating (numeric, 0–5 scale); parsed from
        "X.X/5" string format; "NEW" and "-" treated as missing.
Features: address, name (text), online_order, book_table (categorical),
          votes (numeric), phone, location, rest_type, dish_liked,
          cuisines (text/categorical), approx_cost(for two people) (numeric),
          reviews_list, menu_item (text), listed_in(type), listed_in(city)
          (categorical).
Dropped: url.

Produces:
    <output_dir>/
        data.csv          features + target
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_ZOMATO_RESTAURANTS"
SLUG_BASE = "multabench-zomato-restaurants"
KAGGLE_SOURCE = "himanshupoddar/zomato-bangalore-restaurants"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_FOOD_ZOMATO_RESTAURANTS)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
