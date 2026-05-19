"""
Dataset Name: MUL_TEXT_SPOTIFY_GENRES
====
Examples: 114000
====
URL: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
====
Target Variable: track_genre (object, 114 distinct): ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass', 'blues', 'brazil']
====
Features:

artists (object, 31437 distinct, 0.0% missing): ['The Beatles', 'George Jones', 'Stevie Wonder', 'Linkin Park', 'Ella Fitzgerald', 'Prateek Kuhad', 'Feid', 'Chuck Berry', 'Håkan Hellström', 'OneRepublic']
album_name (object, 46589 distinct, 0.0% missing): ['Alternative Christmas 2022', 'Feliz Cumpleaños con Perreo', 'Metal', 'Halloween con perreito', 'Halloween Party 2022', 'The Complete Hank Williams', 'Fiesta portatil', 'Frescura y Perreo', 'Esto me suena a Farra', 'Perreo en Halloween']
track_name (object, 73608 distinct, 0.0% missing): ['Run Rudolph Run', 'Halloween', 'Frosty The Snowman', 'Little Saint Nick - 1991 Remix', 'Last Last', 'Christmas Time', 'CÓMO SE SIENTE - Remix', 'Sleigh Ride', 'RUMBATÓN', 'X ÚLTIMA VEZ']
popularity (int64, 101 distinct): ['0', '22', '21', '44', '1', '23', '20', '43', '45', '41']
duration_ms (int64, 50697 distinct): ['162897', '180000', '192000', '240000', '118840', '172342', '227520', '131733', '243057', '175986']
explicit (bool, 2 distinct): ['0', '1']
danceability (float64, 1174 distinct): ['0.647', '0.609', '0.579', '0.685', '0.602', '0.524', '0.689', '0.598', '0.607', '0.626']
energy (float64, 2083 distinct): ['0.876', '0.937', '0.931', '0.801', '0.886', '0.948', '0.961', '0.858', '0.92', '0.981']
key (int64, 12 distinct): ['7', '0', '2', '9', '1', '5', '11', '4', '6', '10']
loudness (float64, 19480 distinct): ['-5.662', '-4.457', '-9.336', '-7.57', '-4.034', '-8.871', '-3.725', '-4.324', '-5.08', '-12.472']
mode (int64, 2 distinct): ['1', '0']
speechiness (float64, 1489 distinct): ['0.0323', '0.0324', '0.0322', '0.0328', '0.0295', '0.0321', '0.033', '0.0367', '0.0326', '0.0363']
acousticness (float64, 5061 distinct): ['0.995', '0.993', '0.994', '0.992', '0.991', '0.131', '0.881', '0.108', '0.107', '0.99']
instrumentalness (float64, 5346 distinct): ['0.0', '0.0', '0.905', '0.895', '0.934', '0.922', '0.911', '0.0001', '0.913', '0.9']
liveness (float64, 1722 distinct): ['0.108', '0.111', '0.109', '0.11', '0.105', '0.107', '0.103', '0.106', '0.112', '0.113']
valence (float64, 1790 distinct): ['0.961', '0.304', '0.717', '0.962', '0.324', '0.963', '0.55', '0.365', '0.949', '0.464']
tempo (float64, 45653 distinct): ['0.0', '151.925', '95.004', '130.594', '87.925', '125.004', '92.988', '76.783', '77.321', '90.04']
time_signature (int64, 5 distinct): ['4', '3', '5', '1', '0']
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
