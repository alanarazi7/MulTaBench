"""
Source: Kaggle — verracodeguacas/clear-corpus
        CLEAR corpus: ~4,700 text excerpts annotated for readability.

Target: New Dale-Chall Readability Formula — readability score (numeric).
Features: Author, Title (text), Anthology, Source, Pub Year, Category,
          Location, License (categorical), Excerpt (long text),
          BT Easiness, Flesch-Reading-Ease, Flesch-Kincaid-Grade-Level,
          Automated Readability Index, SMOG Readability, CAREC, CAREC_M,
          CARES, CML2RI (numeric readability indices).
Dropped: Kaggle split, firstPlace_pred through sixthPlace_pred, ID,
         Last Changed.

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


DATASET_ID = "REG_TEXT_BOOK_READABILITY"
SLUG_BASE = "multabench-book-readability"
KAGGLE_SOURCE = "verracodeguacas/clear-corpus"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_SOCIAL_BOOK_READABILITY_CLEAR)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
