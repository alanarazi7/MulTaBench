"""
Source: Kaggle — maharshipandya/-spotify-tracks-dataset
        Spotify tracks with audio features and genre labels.

Target: track_genre — 114-class genre.
Features: artists, album_name, track_name (text), popularity (numeric),
          duration_ms, explicit (binary), danceability, energy, key,
          loudness, mode, speechiness, acousticness, instrumentalness,
          liveness, valence, tempo, time_signature (numeric).
Dropped: Unnamed: 0, track_id.

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


DATASET_ID = "MUL_TEXT_SPOTIFY_GENRES"
SLUG_BASE = "multabench-spotify-genres"
KAGGLE_SOURCE = "maharshipandya/-spotify-tracks-dataset"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.MUL_TEXT_SOCIAL_SPOTIFY_GENRES)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
