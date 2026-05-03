"""
Source: OpenML dataset 46655 (fake_job_postings2)
        Originally from Kaggle: shivamb/real-or-fake-fake-jobposting-prediction

12,725 job postings; predict whether a posting is fraudulent.

Target: fraudulent — binary (0 = real, 1 = fake).
Features: title, salary_range, description (text), required_experience,
          required_education (categorical).

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


DATASET_ID = "BIN_TEXT_FAKE_JOB_POSTING"
SLUG_BASE = "multabench-fake-job-posting"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46655"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.BIN_TEXT_PROFESSIONAL_FAKE_JOB_POSTING)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
