"""
Source: SCImago Journal & Country Rank (2024)
        https://www.scimagojr.com/journalrank.php

Academic journal impact metrics for ~21,000 journals.

Target: H index — journal h-index (numeric).
Features: Title (text), Type, Issn, SJR Best Quartile (categorical),
          SJR, Total Docs, Total Refs, Total Cites, Citable Docs,
          Cites/Doc, Ref/Doc, %Female, Overton, SDG (numeric),
          Country, Region, Publisher (text/categorical),
          Coverage, Categories, Areas (categorical).
Dropped: Sourceid (identifier), Rank (computed from H index).

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


DATASET_ID = "REG_TEXT_SCIMAGOJR_IMPACT"
SLUG_BASE = "multabench-scimagojr-impact"
KAGGLE_SOURCE = "https://www.scimagojr.com/journalrank.php?out=xls"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(UrlDatasetID.REG_TEXT_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
