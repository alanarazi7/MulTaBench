"""
Source: OpenML dataset 46668 (kick_starter_funding)

Kickstarter crowdfunding campaigns; predict whether a campaign was funded.

Target: Funding Status — binary (0 = failed, 1 = successful); renamed from
        raw "final_status".
Features: name, desc, keywords (text), goal (numeric), deadline,
          created_at (date).

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


DATASET_ID = "BIN_TEXT_KICKSTARTER_FUNDING"
SLUG_BASE = "multabench-kickstarter-funding"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46668"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.BIN_TEXT_PROFESSIONAL_KICKSTARTER_FUNDING)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
