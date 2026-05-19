"""
Dataset Name: REG_TEXT_VANCOUVER_SALARIES
====
Examples: 44574
====
Target Variable: remuneration (float64, 39677 distinct): ['139995.02', '126456.0', '114675.28', '110903.0', '110103.0', '97104.0', '87968.0', '124237.0', '102840.0', '97732.0']
====
Features:

year (int64, 17 distinct): ['2024', '2023', '2022', '2021', '2020', '2019', '2018', '2017', '2015', '2016']
name (object, 8308 distinct): ['Wong, B', 'Lee, D', 'Lee, J', 'Ng, W', 'Robinson, D', 'Baker, M', 'Lee, C', 'Lee, M', 'Brown, L', 'Chan, C']
department (object, 31 distinct): ['Engineering Services', 'Fire and Rescue Services', 'VFRS & OEM', 'Board of Parks & Recreation', 'VFRS', 'IT, Digital Strategy & 311', 'Dev Svcs, Bldg & Licensing', 'Community Services', 'Planning, Urban Des & Sustain', 'Vancouver Public Library Board']
title (object, 3118 distinct): ['Firefighter', 'Fire Lieutenant', 'Fire Captain', 'Superintendent I', 'Trades - Electrician', 'FIREFIGHTER', 'Journeyman - Mechanic', 'Civil Engineer I', 'Civil Engineer Ii', 'District Building Inspector']
expenses (float64, 7458 distinct, 5.4% missing): ['0.0', '399.0', '998.0', '561.75', '650.0', '473.0', '3086.0', '105.0', '546.0', '20.0']
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
