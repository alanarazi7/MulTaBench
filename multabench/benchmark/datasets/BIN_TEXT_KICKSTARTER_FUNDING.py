"""
Dataset Name: BIN_TEXT_KICKSTARTER_FUNDING
====
Examples: 86502
====
URL: https://www.openml.org/search?type=data&id=46668
====
Target Variable: Funding Status (object, 2 distinct): ['Failed', 'Successful']
====
Features:

name (object, 86313 distinct, 0.0% missing): ['"New EP/Music Development"', '"Canceled (Canceled)"', '"Aftermath"', '"Broken"', '"New Album"', '"Bunny"', '"Debut Album"', '"Alone"', '"The Journey"', '"Manifest Destiny"']
desc (object, 86005 distinct, 0.0% missing): ['"The Decentralized Dance Party was founded on the belief that Partying is an art form that has the power to change the world."', '"."', '"Breakout Artist Management has offered to work with and develop this project in the studio and we need your help!"', '"Rock Steady is the first manga about Rock. Rock Steady is an action/adventure manga, join us on this extraordinary journey"', '"The Impossible Girl redefines the rock tour. One show at a time."', '"Opera SmackDown turns the traditional vocal concert on its ear by combining elements of professional wrestling and Opera Competitions"', '"After being blackmailed by her evil stepfather, a young woman must frame the man she has fallen for in order to avoid imprisonment."', '"Practically all of the original Dungeons & Dragons artwork that I created during my time at TSR was destroyed. Let\'s bring it back!"', '"A group of American artists exchanging creative education with African artists!"', '"imagine roaming the world’s largest ocean year after year alone, calling out with the regularity of a metronome, & hearing no response."']
goal (float64, 3060 distinct): ['5000.0', '10000.0', '1000.0', '3000.0', '2000.0', '2500.0', '15000.0', '500.0', '1500.0', '20000.0']
keywords (object, 86502 distinct): ['"cross-eyed-chicks"', '"getting-boys-to-read-quick-tips-for-parents-and-te"', '"the-next-page-feature-film"', '"a-drink-sorority-girls-will-love"', '"flickit-with-friends"', '"odie-vice-buy-one-supply-one"', '"the-grizzly"', '"be-a-part-of-the-state-choir-of-russias-return-to"', '"mytotaltv-share"', '"the-milling-gowns-something-dangerous-loves-me-ep"']
disable_communication (bool, 2 distinct): ['0', '1']
country (object, 10 distinct): ['"US"', '"GB"', '"CA"', '"AU"', '"NL"', '"NZ"', '"SE"', '"DK"', '"IE"', '"NO"']
currency (object, 9 distinct): ['"USD"', '"GBP"', '"CAD"', '"AUD"', '"EUR"', '"NZD"', '"SEK"', '"DKK"', '"NOK"']
deadline (object, 81433 distinct): ['1970-01-01 00:00:01.414814340', '1970-01-01 00:00:01.420088340', '1970-01-01 00:00:01.325393940', '1970-01-01 00:00:01.430452740', '1970-01-01 00:00:01.425185940', '1970-01-01 00:00:01.351742340', '1970-01-01 00:00:01.409543940', '1970-01-01 00:00:01.427860740', '1970-01-01 00:00:01.330577940', '1970-01-01 00:00:01.412135940']
created_at (object, 86460 distinct): ['1970-01-01 00:00:01.342830582', '1970-01-01 00:00:01.419183243', '1970-01-01 00:00:01.421797410', '1970-01-01 00:00:01.412122722', '1970-01-01 00:00:01.333308811', '1970-01-01 00:00:01.427162930', '1970-01-01 00:00:01.404826263', '1970-01-01 00:00:01.410551952', '1970-01-01 00:00:01.375932774', '1970-01-01 00:00:01.407786431']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import OpenMLDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "BIN_TEXT_KICKSTARTER_FUNDING"
SLUG_BASE = "multabench-kickstarter-funding"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46668"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.BIN_TEXT_PROFESSIONAL_KICKSTARTER_FUNDING)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
