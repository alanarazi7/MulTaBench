from os.path import join
from typing import List

from pandas import DataFrame, read_csv, Series

from multabench.utils.file_downloading import download_url_image_column
from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: airbnb/seattle/
====
Examples: 3818
====
URL: https://www.kaggle.com/airbnb/seattle
====
Bag Of Tricks Paper:
seattle: Predict the price of living in a hotel based on its image and metadata including text
description, review from customers, room type and so on. This dataset originally stems from
https://www.kaggle.com/datasets/airbnb/seattle. We randomly split the original training set at
4:1 ratio for new training and test sets. The license of the original dataset: CC0 1.0.

Description:
Airbnb listings from Seattle with structured features such as property type, room type, coordinates,
and host metrics, alongside rich textual fields like listing summaries, neighborhood overviews, and
amenities.
====
Target Variable: price (float64, 273 distinct): ['150.0', '100.0', '75.0', '95.0', '99.0', '90.0']
====
Features:

name (object, 3792 distinct): listing name
summary (object, 3478 distinct): listing summary
space (object, 3119 distinct): space description
description (object, 3742 distinct): full listing description
neighborhood_overview (object, 2506 distinct): neighborhood description
notes (object, 1999 distinct): host notes
house_image (object, 2864 distinct): URL of house photo
[many numeric and categorical columns for property type, bedrooms, review scores, etc.]
'''


def load_df(dir_path: str) -> DataFrame:
    df_path = join(dir_path, 'listings.csv')
    df = read_csv(df_path)
    img_folder = join(dir_path, IMAGE_FOLDER)
    df = _get_house_image(df=df, img_folder=img_folder)
    return df


def _get_house_image(df: DataFrame, img_folder: str) -> DataFrame:
    house_img_cols = ['picture_url', 'thumbnail_url', 'medium_url', 'xl_picture_url']
    for col in house_img_cols:
        df = download_url_image_column(df=df, img_folder=img_folder, img_col=col)
    df['house_image'] = df.apply(lambda r: _take_first_pic(r, cols=house_img_cols), axis=1)
    df.drop(columns=house_img_cols, inplace=True)
    return df


def _take_first_pic(row: Series, cols: List[str]) -> str:
    for c in cols:
        v = row[c]
        if v:
            return v
    return None


def _extract_price(s: str) -> float:
    if isinstance(s, float):
        return s
    try:
        s = s.replace('$', '').replace(',', '')
        return float(s)
    except Exception:
        return None


def _extract_percentage(s: str) -> float:
    if isinstance(s, float):
        return s
    try:
        return float(s.replace('%', ''))
    except Exception:
        return None


CONTEXT = "Predicting Airbnb listing price in Seattle from house image, text descriptions, and property metadata"
TARGET = CuratedTarget(raw_name="price", task_type=SupervisedTask.REGRESSION, processing_func=_extract_price)
FEATURES = [
    CuratedFeature(raw_name='house_image', feat_type=FeatureType.IMAGE),
    CuratedFeature(raw_name='name', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='summary', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='space', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='description', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='neighborhood_overview', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='notes', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='first_review', feat_type=FeatureType.DATE),
    CuratedFeature(raw_name='last_review', feat_type=FeatureType.DATE),
    CuratedFeature(raw_name='host_since', feat_type=FeatureType.DATE),
    CuratedFeature(raw_name='host_response_rate', processing_func=_extract_percentage),
    CuratedFeature(raw_name='host_acceptance_rate', processing_func=_extract_percentage),
]
COLS_TO_DROP = [
    'id', 'host_id',
    'calendar_last_scraped', 'has_availability', 'scrape_id', 'last_scraped', 'experiences_offered',
    'market', 'country_code', 'country', 'requires_license', 'license', 'jurisdiction_names',
    'listing_url', 'host_url',
    'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee', 'extra_people',
    # host_image dropped — only house_image used for BagOfTricks parity (#Img=1)
    'host_picture_url', 'host_thumbnail_url',
]
IMAGE_FOLDER = "downloaded_images"
LOADING_FUNC = load_df
