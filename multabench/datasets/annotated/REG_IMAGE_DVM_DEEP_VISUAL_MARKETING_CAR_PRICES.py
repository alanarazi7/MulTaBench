import os
from os.path import exists, join
from typing import Any, Optional

from pandas import DataFrame

from multabench.datasets.utils import load_csv
from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType


IMAGE_FEATURE_NAME = "Car Picture"

'''
Dataset Name: osamasaifs/dvm-car-with-images/
====
Examples: 246159
====
URL: https://www.kaggle.com/osamasaifs/dvm-car-with-images
====
Bag Of Tricks Paper:
DVM: Predict the selling price of cars based on car images and the metadata including fuel type,
the number of seats, body type and so on. This dataset originally stems from [36]: https://
deepvisualmarketing.github.io. The original dataset contains multiple metadata tables. We
use Ad table as the metadata, which contains more than 0.25 million used car advertisements. We
further select 25% of the metadata randomly, and split the subset at 3:1 ratio for a new training set
and test set. The original dataset provide car images with different views, and we use the front view.
The license of the original dataset: CC BY-NC 4.0.

Description:
This dataset is based on the DVM-Car (Deep Visual Marketing - Car) dataset, a large-scale, publicly
available dataset designed to support automotive industry research and applications.
====
Target Variable: Price (float64, 19741 distinct): ['3995.0', '5995.0', '4995.0', '2995.0', '6995.0']
====
Features:

Maker (object, 84 distinct): ['Ford', 'Audi', 'Vauxhall', 'Volkswagen', 'BMW', 'Nissan', 'Peugeot']
Genmodel (object, 873 distinct): ['Corsa', 'Focus', 'Fiesta', 'Juke', 'X5', 'Astra', 'Mondeo']
Adv_year (int64, 10 distinct): ['2018', '2021', '2017', '2020', '2016', '2019', '2015', '2014']
Adv_month (int64, 15 distinct): ['5', '8', '4', '7', '6', '3', '2', '1', '12', '11']
Color (object, 22 distinct, 8.2% missing): ['Black', 'Silver', 'Blue', 'Grey', 'White', 'Red']
Reg_year (float64, 25 distinct): ['2015.0', '2017.0', '2016.0', '2014.0', '2018.0', '2013.0']
Bodytype (object, 16 distinct, 0.3% missing): ['Hatchback', 'SUV', 'MPV', 'Saloon', 'Coupe', 'Estate']
Runned_Miles (object, 69651 distinct, 0.3% missing): ['10', '100000', '80000', '60000', '70000']
Engin_size (object, 70 distinct, 0.8% missing): ['2.0L', '1.6L', '3.0L', '1.2L', '1.4L', '1.0L']
Gearbox (object, 3 distinct, 0.1% missing): ['Manual', 'Automatic', 'Semi-Automatic']
Fuel_type (object, 13 distinct, 0.1% missing): ['Diesel', 'Petrol', 'Hybrid  Petrol/Electric', 'Electric']
Seat_num (float64, 10 distinct, 2.3% missing): ['5.0', '4.0', '7.0', '2.0', '8.0', '6.0']
Door_num (float64, 6 distinct, 1.6% missing): ['5.0', '3.0', '4.0', '2.0', '6.0', '7.0']
Predicted_viewpoint (int64, 9 distinct): ['0', '225', '90', '45', '315', '270', '180', '135', '360']
Quality_check (object, 2 distinct, 74.3% missing): ['P', 'N']
Car Picture (object, 246159 distinct): ['Bentley/Arnage/2000/Silver/Bentley$$Arnage$$2000$$Silver$$10_1$$1$$image_0.jpg', ...]
'''


def load_df(dir_path: str) -> DataFrame:
    tables_dir = join(dir_path, NUM_DIR, "tables", "tables")
    img_df = _get_img_df(tables_dir)
    ad_df = load_csv(tables_dir, "Ad_table.csv")
    df = ad_df.merge(img_df, on='Adv_ID', how='inner')
    df[IMAGE_FEATURE_NAME] = df["Image_name"].apply(lambda x: _find_image_path(image_name=x, dir_path=dir_path))
    df.columns = [c.strip() for c in df.columns]
    df.drop(columns=['Adv_ID', 'Genmodel_ID', 'Image_ID', 'Image_name'], inplace=True)
    return df


def _get_img_df(tables_dir: str) -> DataFrame:
    img_df = load_csv(tables_dir, "Image_table.csv")
    img_df.columns = [c.strip() for c in img_df.columns]
    img_df["Adv_ID"] = img_df["Image_ID"].str.rsplit("$$", n=1).str[0]
    img_df = img_df.groupby("Adv_ID").first().reset_index()
    return img_df


def _find_image_path(image_name: str, dir_path: str) -> str:
    brand_model_year_color = "/".join(image_name.split("$$")[:4])
    image_path = join(brand_model_year_color, image_name).replace("'", "")
    full_path = join(dir_path, IMAGE_FOLDER, image_path)
    if not exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    return image_path


def _normalize_price(price: Any) -> Optional[float]:
    if isinstance(price, (int, float)):
        return price
    if isinstance(price, str) and price.isdigit():
        return int(price)
    if 'ukn' in price.lower() or 'unk' in price.lower():
        return None
    raise ValueError(f"Cannot normalize price: {price}")


CONTEXT = "Predicting car selling price from front-view images and ad metadata"
TARGET = CuratedTarget(raw_name="Price", task_type=SupervisedTask.REGRESSION, processing_func=_normalize_price)
COLS_TO_DROP = []
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
NUM_DIR = "19586296"
IMAGE_FOLDER = "19586296/resized_DVM_clean/resized_DVM_clean"
LOADING_FUNC = load_df
