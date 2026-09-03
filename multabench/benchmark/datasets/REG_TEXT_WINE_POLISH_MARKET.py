"""
Source: https://www.kaggle.com/datasets/skamlo/wine-price-on-polish-market

Wines on the Polish market; predict the price.

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


DATASET_ID = "REG_TEXT_WINE_POLISH_MARKET"
SLUG_BASE = "multabench-full-wine-polish-market"
KAGGLE_SOURCE = "https://www.kaggle.com/datasets/skamlo/wine-price-on-polish-market"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_FOOD_WINE_POLISH_MARKET_PRICES)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate REG_TEXT_WINE_POLISH_MARKET for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
