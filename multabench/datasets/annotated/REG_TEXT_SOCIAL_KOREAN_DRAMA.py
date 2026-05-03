from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedFeature, CuratedTarget
from multabench.datasets.objects import SupervisedTask, FeatureType
from multabench.datasets.utils import load_csv

'''
Dataset Name: noorrizki/top-korean-drama-list-1500/
====
Examples: 1647
====
URL: https://www.kaggle.com/noorrizki/top-korean-drama-list-1500
====
Description:
Top Korean Drama List (~1500)
This dataset contains 1646 data top kdrama's from MyDramalist (Apr 2023)

About Dataset
This dataset contains information about 1,646 Korean dramas obtained from web scraping the website https://mydramalist.com/ in April 2023. The dataset consists of 1646 rows and 10 columns: Title, Year, Score, Synopsis, URL, Cast, Rating, Network, Genre, and Tags. The dataset can be used for various analysis and research purposes related to Korean dramas.

Acknowledgements
This data is taken from the website https://mydramalist.com/shows/top?page=1 , I would like to express our gratitude to MyDramaList.com for providing a comprehensive source of information for Kdrama enthusiasts. Their platform has been instrumental in helping me compile this dataset.

====
Target Variable: Score (float64, 28 distinct): ['7.4', '7.5', '7.3', '7.9', '7.8', '7.6', '7.7', '7.2', '8.1', '7.1']
====
Features:

Name (object, 1642 distinct): ['Hyena', 'Crazy Love', 'Hero', 'Trap', 'Once Again', ...]
Year (int64, 26 distinct): ['2022', '2019', '2021', '2020', '2018', ...]
Genre (object, 584 distinct): ['Comedy, Romance, Drama', 'Romance, Drama, Melodrama', ...]
Main Cast (object, 1639 distinct): ['Yoo Ji Tae, Park Hae Soo, Jeon Jong Seo, ...']
Sinopsis (object, 1640 distinct, 0.3% missing): [...]
Content Rating (object, 6 distinct): ['15+ - Teens 15 or older', 'Not Yet Rated', '13+ - Teens 13 or older', ...]
Tags (object, 1622 distinct, 1.2% missing): ['Soap Opera', 'Adapted From A Manhwa', 'Miniseries', ...]
Network (object, 373 distinct): ['Viki', 'Netflix', 'Apple TV, Viki', ...]
Number of Episodes (float64, 97 distinct): ['16.0', '20.0', '12.0', '8.0', '10.0', ...]
'''


def get_episodes(episode: str) -> int:
    episode = episode.lower()
    assert episode.endswith("episodes"), f"Invalid episode format: {episode}"
    episode = episode.replace("episodes", "").strip()
    return int(episode)


def load_df(dir_path: str) -> DataFrame:
    return load_csv(dir_path, "kdrama_list.csv")


CONTEXT = "Korean Dramas"
TARGET = CuratedTarget(raw_name="Score", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["Unnamed: 0",
                # Image poster URLs — not usable as local paths in TEXT version
                "img url"]
FEATURES = [CuratedFeature(raw_name="Episode", new_name="Number of Episodes",
                           processing_func=get_episodes, feat_type=FeatureType.NUMERIC)]
LOADING_FUNC = load_df
