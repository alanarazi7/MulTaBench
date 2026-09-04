"""
Source: https://www.kaggle.com/selener/consumer-complaint-database

1,282,348 rows. Consumer complaints filed against financial companies.

Target: Company response to consumer — 8 classes (Closed with explanation, Closed with
        monetary relief, In progress, ...).
Features: Product, Sub-product, Issue, Consumer complaint narrative (text), Company,
          State, Submitted via, plus two date columns.

The source enum is named BIN_TEXT_* but the target has 8 classes, so the MulTaBench
name uses the MUL_ prefix.

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


DATASET_ID = "MUL_TEXT_CONSUMER_COMPLAINT"
SLUG_BASE = "multabench-full-consumer-complaint"
KAGGLE_SOURCE = "https://www.kaggle.com/selener/consumer-complaint-database"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.BIN_TEXT_FINANCIAL_CONSUMER_COMPLAINT)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate MUL_TEXT_CONSUMER_COMPLAINT for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
