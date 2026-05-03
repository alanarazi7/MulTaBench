"""
Source: Vancouver Open Data
        https://opendata.vancouver.ca — employee remuneration and expenses
        for employees earning over $75,000.

Target: remuneration — annual remuneration in CAD (numeric).
Features: year (numeric), name (text), department, title (categorical),
          expenses (numeric).

Produces:
    <output_dir>/
        data.csv          features + target
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import UrlDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_VANCOUVER_SALARIES"
SLUG_BASE = "multabench-vancouver-salaries"
KAGGLE_SOURCE = "https://opendata.vancouver.ca/api/records/1.0/download/?dataset=employee-remuneration-and-expenses-earning-over-75000&format=csv"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(UrlDatasetID.REG_TEXT_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
