from ast import literal_eval
from typing import Any, Dict, Optional, List

import numpy as np
from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedFeature, CuratedTarget
from multabench.datasets.objects import SupervisedTask, FeatureType
from multabench.datasets.utils import load_csv

'''
Dataset Name: rounakbanik/the-movies-dataset/movies_metadata.csv
====
Examples: 45466
====
URL: https://www.kaggle.com/rounakbanik/the-movies-dataset/movies_metadata.csv
====
Description:
The Movies Dataset
Metadata on over 45,000 movies. 26 million ratings from over 270,000 users.

About Dataset
Context
These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages.

This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.

Content
This dataset consists of the following files:

movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.

keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.

credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.

links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.

links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.

ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.

The Full MovieLens Dataset consisting of 26 million ratings and 750,000 tag applications from 270,000 users on all the 45,000 movies in this dataset can be accessed here

Acknowledgements
This dataset is an ensemble of data collected from TMDB and GroupLens.
The Movie Details, Credits and Keywords have been collected from the TMDB Open API. This product uses the TMDb API but is not endorsed or certified by TMDb. Their API also provides access to data on many additional movies, actors and actresses, crew members, and TV shows. You can try it for yourself here.

The Movie Links and Ratings have been obtained from the Official GroupLens website. The files are a part of the dataset available here

====
Target Variable: revenue (float64, 6863 distinct): ['0.0', '12000000.0', '11000000.0', '10000000.0', '2000000.0', '6000000.0', '5000000.0', '8000000.0', '500000.0', '1.0']
====
Features:

adult (object, 2 distinct): ['False', 'True']
collection (object, 1696 distinct): ['', 'The Bowery Boys', 'Totò Collection', 'James Bond Collection']
budget (object, 1223 distinct): ['0', '5000000', '10000000', '20000000', '2000000', '15000000', '3000000', '25000000', '1000000', '30000000']
genres (object, 1921 distinct): ['Drama', 'Comedy', 'Documentary', '', 'Comedy, Drama', 'Drama, Romance', 'Comedy, Romance']
original_language (object, 89 distinct): ['en', 'fr', 'it', 'ja', 'de', 'es', 'ru', 'hi', 'ko', 'zh']
original_title (object, 43369 distinct): ['Alice in Wonderland', 'Hamlet', 'Macbeth', 'The Three Musketeers', 'Les Misérables', 'A Christmas Carol']
overview (object, 44303 distinct): ['No overview found.', 'No Overview', ' ']
popularity (object, 44175 distinct): ['0.0', '0.0', '1e-06', '0.0', '0.0008', '0.0006', '0.00022', '0.0003', '0.0012', '0.000308']
production_companies (object, 22666 distinct): ['', 'Metro-Goldwyn-Mayer (MGM)', 'Warner Bros.', 'Paramount Pictures', 'Twentieth Century Fox Film Corporation']
production_countries (object, 1832 distinct): ['United States of America', '', 'United Kingdom', 'France', 'Japan', 'Italy', 'Canada', 'Germany', 'Russia', 'India']
release_date (datetime64[ns], 0 distinct): ['2008-01-01 00:00:00', '2009-01-01 00:00:00', '2007-01-01 00:00:00']
runtime (float64, 353 distinct): ['90.0', '0.0', '100.0', '95.0', '93.0', '96.0', '92.0', '94.0', '91.0', '88.0']
spoken_languages (object, 1428 distinct): ['English', '', 'Français', '日本語', 'Italiano', 'Español', 'English,Français']
status (object, 6 distinct): ['Released', 'Rumored', 'Post Production', 'In Production', 'Planned', 'Canceled']
tagline (object, 20283 distinct): ['Based on a true story.', '-', 'Be careful what you wish for.', 'Trust no one.', 'Know Your Enemy']
title (object, 42277 distinct): ['Cinderella', 'Alice in Wonderland', 'Hamlet', 'Les Misérables', 'Beauty and the Beast']
video (object, 2 distinct): ['0', '1']
vote_average (float64, 92 distinct): ['0.0', '6.0', '5.0', '7.0', '6.5', '6.3', '5.5', '5.8', '6.4', '6.7']
vote_count (float64, 1820 distinct): ['1.0', '2.0', '0.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
'''


def extract_genre(g: str) -> str:
    # Rows look like: "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]"
    g = literal_eval(g)
    genres = sorted([d['name'] for d in g])
    genres = ", ".join(genres)
    return genres

def extract_language(lang: Any) -> str:
    # Rows look like: "[{'iso_3166_1': 'US', 'name': 'United States of America'}]"
    return extract_from_named_lists(lang)

def extract_collection(collect: Any) -> str:
    # Rows look like: {'id': 10194, 'name': 'Toy Story Collection', 'poster_path': '/7G9915LfUQ2lVfwMEEhDsn3kT4B.jpg', 'backdrop_path': '/9FBwqcd9IRruEDUrTdcaafOMKUq.jpg'}
    d_collect = normalize_raw_json(collect)
    if d_collect is None:
        return ""
    return d_collect['name']


def extract_from_named_lists(raw: Any) -> Any:
    evaluated = normalize_raw_json(raw)
    if evaluated is None:
        return ""
    names = sorted([d['name'] for d in evaluated])
    final_names = ",".join(names)
    return final_names

def normalize_raw_json(raw: Any) -> Optional[Dict | List]:
    if raw is np.nan:
        return None
    evaluated = literal_eval(raw)
    if isinstance(evaluated, (float, str, bool)):
        return None
    return evaluated


def load_df(dir_path: str) -> DataFrame:
    return load_csv(dir_path, "movies_metadata.csv")


CONTEXT = "Metadata of movies released until 2020 for box-office revenues"
TARGET = CuratedTarget(raw_name="revenue", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["id", "imdb_id", "poster_path", "homepage",]
FEATURES = [
    CuratedFeature(raw_name="release_date", feat_type=FeatureType.DATE),
    CuratedFeature(raw_name="genres", processing_func=extract_genre),
    CuratedFeature(raw_name="spoken_languages", processing_func=extract_language),
    CuratedFeature(raw_name="belongs_to_collection", new_name="collection", processing_func=extract_collection),
    CuratedFeature(raw_name="production_companies", processing_func=extract_from_named_lists),
    CuratedFeature(raw_name="production_countries", processing_func=extract_from_named_lists),
]
LOADING_FUNC = load_df

DESCRIPTION = '''
The Movies Dataset
Metadata on over 45,000 movies. 26 million ratings from over 270,000 users.

About Dataset
Context
These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages.

This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.

Content
This dataset consists of the following files:

movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.

keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.

credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.

links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.

links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.

ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.

The Full MovieLens Dataset consisting of 26 million ratings and 750,000 tag applications from 270,000 users on all the 45,000 movies in this dataset can be accessed here

Acknowledgements
This dataset is an ensemble of data collected from TMDB and GroupLens.
The Movie Details, Credits and Keywords have been collected from the TMDB Open API. This product uses the TMDb API but is not endorsed or certified by TMDb. Their API also provides access to data on many additional movies, actors and actresses, crew members, and TV shows. You can try it for yourself here.

The Movie Links and Ratings have been obtained from the Official GroupLens website. The files are a part of the dataset available here
'''
