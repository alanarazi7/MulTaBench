"""
Source: Kaggle — sobhanmoosavi/us-accidents
        US traffic accident records up to March 2023 (7.7M rows).

Target: Severity — 4-class (1–4 severity level).
Features: Source, location (lat/lng, street, city, state, zipcode),
          Description (text), weather conditions (temperature, humidity,
          pressure, visibility, wind, precipitation, weather condition text),
          11 boolean road feature flags (bump, crossing, junction, etc.),
          daylight/twilight indicators, timestamps.
Dropped: ID.

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


DATASET_ID = "MUL_TEXT_US_ACCIDENTS"
SLUG_BASE = "multabench-us-accidents"
KAGGLE_SOURCE = "sobhanmoosavi/us-accidents"

SAMPLE_SIZE = 100_000



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.MUL_TEXT_TRANSPORTATION_US_ACCIDENTS_MARCH23)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    if len(df) > SAMPLE_SIZE:
        df = df.groupby(dataset.y.name, group_keys=False).apply(
            lambda g: g.sample(frac=SAMPLE_SIZE / len(df), random_state=42)
        ).reset_index(drop=True)
        print(f"  Sampled to {len(df):,} rows (stratified)")
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
