from os.path import join

from pandas import DataFrame

from multabench.datasets.utils import load_csv
from multabench.utils.file_downloading import download_url_image_column
from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType

IMAGE_FEATURE_NAME = "img"

'''
Dataset Name: mdhamani/goodreads-books-100k/
====
Examples: 100000
====
URL: https://www.kaggle.com/mdhamani/goodreads-books-100k
====
Bag Of Tricks Paper:
goodreads: Predict the rating of a book based on its cover image and metadata including title, content
descriptions, genre and so on. This dataset originally stems from https://www.kaggle.com/
datasets/mdhamani/goodreads-books-100k. We remove non-English sentences, randomly select 16% of
the original dataset and split this subset at 3:1 ratio for new training and test sets.
The license of the original dataset: CC0 1.0.

Description:
GoodReads 100k books — An extensive GoodReads dataset containing 100k books.
====
Target Variable: rating (float64, 289 distinct): ['4.0', '0.0', '3.67', '3.75', '3.8', '3.88', '3.5']
====
Features:

author (object, 68767 distinct): ['Mi-Ri Hwang', 'Willy Vandersteen', 'Yu-Rang Han', 'R.L. Stine']
bookformat (object, 202 distinct, 3.2% missing): ['Paperback', 'Hardcover', 'ebook', 'Kindle Edition']
desc (object, 92499 distinct, 6.8% missing): [long book descriptions ...]
genre (object, 72129 distinct, 10.5% missing): ['Nonfiction', 'History', 'Fiction', 'Poetry']
img (object, 96955 distinct, 3.0% missing): ['https___i.gr-assets.com_...jpg', ...]
pages (int64, 1357 distinct): ['0', '192', '224', '32', '256', '288']
reviews (int64, 2950 distinct): ['0', '1', '2', '3', '4', '5', '6', '7']
title (object, 97588 distinct, 0.0% missing): ['Love in the Mask', 'Selected Poems', 'Cinderella']
totalratings (int64, 10536 distinct): ['1', '0', '2', '3', '4', '5', '6', '8']
'''


def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "GoodReads_100k_books.csv")
    img_folder = join(dir_path, IMAGE_FOLDER)
    df = download_url_image_column(df=df, img_folder=img_folder, img_col=IMAGE_FEATURE_NAME)
    df.drop(columns=['isbn', 'isbn13', 'link'], inplace=True)
    df = df[df[IMAGE_FEATURE_NAME] != ""]
    return df


CONTEXT = "Predicting book rating from cover image and metadata"
TARGET = CuratedTarget(raw_name="rating", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = [
    CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
    CuratedFeature(raw_name='desc', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='title', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='genre', feat_type=FeatureType.TEXT),
]
IMAGE_FOLDER = "downloaded_images"
LOADING_FUNC = load_df
