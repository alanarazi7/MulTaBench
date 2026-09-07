import os

import pandas as pd

from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import bin_target, save_dataset, task_type_from_name


DATASET_ID = "MUL_TEXT_BOX_OFFICE"
SLUG_BASE = "multabench-full-box-office"
KAGGLE_SOURCE = "https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset"
TARGET_BINS = 5


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_SOCIAL_MOVIES_DATASET_REVENUE)
    x, y = dataset.x, dataset.y
    # 84% of rows record revenue as 0, which is unreported rather than a real gross: keeping them
    # makes the target mostly a constant and collapses every quantile bin into one.
    reported = y > 0
    x, y = x[reported].reset_index(drop=True), y[reported].reset_index(drop=True)
    y = bin_target(y, n_bins=TARGET_BINS)
    df = pd.concat([x, y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID} for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
