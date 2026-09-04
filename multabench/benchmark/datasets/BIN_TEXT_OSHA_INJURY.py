"""
Source: https://www.kaggle.com/ruqaiyaship/osha-accident-and-injury-data-1517

4,847 rows. OSHA workplace accident and injury reports, 2015-2017.

Target: Task Assigned — binary (Regularly Assigned / Not Regularly Assigned).
Features: Free-text accident summary, degree of injury, nature of injury, body part,
          event type, plus the event date.

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


DATASET_ID = "BIN_TEXT_OSHA_INJURY"
SLUG_BASE = "multabench-full-osha-injury"
KAGGLE_SOURCE = "https://www.kaggle.com/ruqaiyaship/osha-accident-and-injury-data-1517"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.BIN_TEXT_TRANSPORTATION_OSHA_ACCIDENT_INJURY_DATA)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate BIN_TEXT_OSHA_INJURY for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
