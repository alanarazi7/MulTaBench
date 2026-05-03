"""
Source: OpenML dataset 46654 (jigsaw_unintended_bias100K)
        100K sample from the Jigsaw Unintended Bias in Toxicity Classification dataset.

Target: Is Toxic — binary (0 = not toxic, 1 = toxic); renamed from raw "target".
Features: comment_text (text), 28 identity attributes (float, ~77% missing),
          engagement metrics (numeric), created_date (date).
Dropped: id, severe_toxicity, toxicity_annotator_count, identity_annotator_count,
         publication_id, parent_id, article_id, rating, insult, obscene, threat,
         identity_attack, sexual_explicit.

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


DATASET_ID = "BIN_TEXT_JIGSAW_TOXICITY"
SLUG_BASE = "multabench-jigsaw-toxicity"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46654"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.BIN_TEXT_SOCIAL_JIGSAW_TOXICITY)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
