"""
Source: OpenML dataset 42125 (employee_salaries)
        Montgomery County employee salary records.

Target: current_annual_salary — annual salary in USD (numeric).
Features: full_name (text), gender (binary), 2016_gross_pay_received,
          2016_overtime_pay (numeric), department, department_name (categorical),
          division, assignment_category, employee_position_title,
          underfilled_job_title (text/categorical), date_first_hired (date).
Dropped: year_first_hired (redundant with date_first_hired).

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


DATASET_ID = "REG_TEXT_MONTGOMERY_SALARIES"
SLUG_BASE = "multabench-montgomery-salaries"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=42125"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.REG_TEXT_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
