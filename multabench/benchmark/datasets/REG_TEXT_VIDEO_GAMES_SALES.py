"""
Source: Kaggle — gregorut/videogamesales
        Video game sales data (~16,600 titles).

Target: Global_Sales — global sales in millions of units (numeric).
Features: Name (text), Platform, Year, Genre, Publisher (categorical).
Dropped: Rank (identifier), NA_Sales, EU_Sales, JP_Sales, Other_Sales
         (leakage — they sum to Global_Sales).

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


DATASET_ID = "REG_TEXT_VIDEO_GAMES_SALES"
SLUG_BASE = "multabench-video-games-sales"
KAGGLE_SOURCE = "gregorut/videogamesales"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_SOCIAL_VIDEO_GAMES_SALES)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
