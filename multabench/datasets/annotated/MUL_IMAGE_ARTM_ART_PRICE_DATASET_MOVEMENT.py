import os
from os.path import exists, join

from pandas import DataFrame

from multabench.datasets.utils import load_csv
from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType

MOVEMENT = "movement"
IMAGE_FEATURE_NAME = "Art Picture"

'''
Dataset Name: flkuhm/art-price-dataset/
====
Examples: 754
====
URL: https://www.kaggle.com/flkuhm/art-price-dataset
====
Bag Of Tricks Paper:
artm: Predict the movement of an artwork (34 total) based on its image and metadata including
title, price, condition and so on. This dataset originally stems from https://www.kaggle.com/
datasets/flkuhm/art-price-dataset. We randomly split the data at 3:1 ratio for new training
set and test set. The license of the original dataset: CC BY-NC-SA 4.0.

Description:
About Dataset
In addition to images of artworks and sculptures that have been offered for sale on Sothebys, the dataset also
contains further information about the associated artworks, such as their price or associated movement.
====
Target Variable: movement (object, 34 distinct): ['Realism', 'Abstract', 'Expressionism', 'Pop Art', 'Conceptual Art', 'Surrealism', 'Impressionism', ...]
====
Features:

price (object, 108 distinct): ['800 USD', '1.500 USD', '680 USD', '3.000 USD', '1.275 USD', '5.000 USD', '6.000 USD', '15.000 USD']
artist (object, 454 distinct): ['Russell Young', 'John Fischer', 'Ruth Bernhard', 'Donald Sultan', 'Richard Bernstein']
title (object, 679 distinct): ['Untitled', 'Untitled ', 'Madame de Pompadour (née Poisson)', 'Rays', 'Fragments Of Hope']
yearCreation (object, 136 distinct): ['2012', '1990', '1989', '2008', '1986', '2021', '[nan]', '2016', '2014', '2011']
signed (object, 390 distinct): ['[nan]', 'Signed lower right', 'Signed verso', 'Signed lower right recto']
condition (object, 376 distinct): ['Excellent condition.', 'The work is in excellent condition, direct from the publisher.']
period (object, 5 distinct): ['Contemporary', 'Post-War', 'Modern', '19th Century', '[nan]']
Art Picture (object, 754 distinct): ['image_1.png', 'image_2.png', 'image_3.png', ...]
'''


def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "artDataset.csv")
    _add_images(df, dir_path=dir_path)
    df = df[df[MOVEMENT].notna()]
    df.drop(columns=['Unnamed: 0'], inplace=True)
    return df


def _add_images(df: DataFrame, dir_path: str):
    df[IMAGE_FEATURE_NAME] = [f"image_{n+1}.png" for n in range(len(df))]
    for img in df[IMAGE_FEATURE_NAME]:
        img_path = join(dir_path, IMAGE_FOLDER, img)
        if not exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")


def fix_price_usd(p: str) -> float:
    # Examples: ['800 USD', '1.500 USD', '680 USD']
    if p.endswith(' USD'):
        p = p[:-4]
    return float(p)


CONTEXT = "Predicting the art movement of an artwork based on its image and metadata"
TARGET = CuratedTarget(raw_name=MOVEMENT, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [
    CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
    CuratedFeature(raw_name='title', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='condition', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='signed', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='price', processing_func=fix_price_usd, feat_type=FeatureType.NUMERIC),
]
IMAGE_FOLDER = "artDataset"
LOADING_FUNC = load_df
