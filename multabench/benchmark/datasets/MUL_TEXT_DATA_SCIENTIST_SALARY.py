"""
Source: OpenML dataset 46664 (data_scientist_salary)
        Data scientist job postings with salary range labels.

Target: salary — 6-class salary bracket (0-3 / 3-6 / 6-10 / 10-15 / 15-25 / 25-50
        lakh INR); mapped from raw codes (e.g. 0to3 → 0-3).
Features: experience, job_description, job_desig, job_type, key_skills,
          location (text/categorical).

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


DATASET_ID = "MUL_TEXT_DATA_SCIENTIST_SALARY"
SLUG_BASE = "multabench-data-scientist-salary"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46664"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.MUL_TEXT_PROFESSIONAL_DATA_SCIENTIST_SALARY)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
