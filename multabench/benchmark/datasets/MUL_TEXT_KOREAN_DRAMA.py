"""
Source: https://www.kaggle.com/datasets/noorrizki/top-korean-drama-list-1500

Top Korean dramas; predict the rating decile.

Target: reformulated from regression into 10 equal-frequency bins at curation time, so the
uploaded artifact carries the binned target -- the bin count is frozen in
multabench/datasets/target_bins.py and the dataset was drawn as a converter by
multabench/datasets/binned_selection.py (stratified draw, seed 20260902).

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


DATASET_ID = "MUL_TEXT_KOREAN_DRAMA"
SLUG_BASE = "multabench-full-korean-drama"
KAGGLE_SOURCE = "https://www.kaggle.com/datasets/noorrizki/top-korean-drama-list-1500"
TARGET_BINS = 10


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_SOCIAL_KOREAN_DRAMA, target_bins=TARGET_BINS)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate MUL_TEXT_KOREAN_DRAMA for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
