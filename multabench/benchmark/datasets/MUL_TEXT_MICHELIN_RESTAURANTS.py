"""
Source: Kaggle — ngshiheng/michelin-guide-restaurants-2021
        Michelin Guide 2021 restaurant listings with location and description.

Target: Award — 5-class (Selected / Bib Gourmand / 1 Star / 2 Stars / 3 Stars).
Features: Name, Address, Location, Price, Cuisine (text/categorical),
          Longitude, Latitude, PhoneNumber (numeric), GreenStar (binary),
          FacilitiesAndServices, Description (text).
Dropped: Url, WebsiteUrl.

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


DATASET_ID = "MUL_TEXT_MICHELIN_RESTAURANTS"
SLUG_BASE = "multabench-michelin-restaurants"
KAGGLE_SOURCE = "ngshiheng/michelin-guide-restaurants-2021"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.MUL_TEXT_FOOD_MICHELIN_GUIDE_RESTAURANTS)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
