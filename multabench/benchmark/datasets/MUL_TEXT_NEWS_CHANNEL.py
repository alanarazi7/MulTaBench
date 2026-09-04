"""
Source: https://www.openml.org/search?type=data&id=46652

20,284 rows. Online news articles; predict the channel that published them.

Target: News Category — 6 classes (World, Tech, Entertainment, Business, Social Media,
        Lifestyle).
Features: Article title and keyword text alongside engagement and structural counts.

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


DATASET_ID = "MUL_TEXT_NEWS_CHANNEL"
SLUG_BASE = "multabench-full-news-channel"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46652"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.MUL_TEXT_SOCIAL_NEWS_CHANNEL_CATEGORY)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate MUL_TEXT_NEWS_CHANNEL for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
