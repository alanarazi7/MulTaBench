"""
Dataset Name: REG_TEXT_BABIES_PRICES
====
Examples: 5085
====
Target Variable: price (float64, 202 distinct): ['19.99', '29.99', '24.99', '14.99', '49.99', '17.99', '34.99', '39.99', '12.99', '59.99']
====
Features:

title (object, 4998 distinct): ['Minene Muslin Squares 2-Pack', "Marmont Hill - 'Zebras 2' Eric Carle Framed Art Print", "Marmont Hill - 'Snoopy All-Star 1950' Peanuts Framed Art Print", "Marmont Hill - 'Yellow Sunflower' Eric Carle Framed Art Print", "Marmont Hill - 'Big Brown Bear 2' Eric Carle Framed Art Print", 'Bacati Circles and Stripes Hamper', 'Minene Muslin Squares 3-Pack', "Marmont Hill - 'Belle and Snoopy' Peanuts Print on Canvas", 'Bacati Quilted Circles Changing Pad Cover', "Baby's Journey Deluxe Pillowtop Changing Pad"]
is_discounted (int64, 2 distinct): ['0', '1']
category (object, 10 distinct): ['Room Decor', 'Nursery Bedding / Blankets', 'Nursery Bedding', 'Storage & Organization', 'Room Decor / Wall Decor', "Kids' Bedding / Twin & Full Bedding", 'Nursery Bedding / Sheets & Pads', "Kids' Bedding / Toddler Bedding", 'Room Decor / Wall Decor / Hanging Letters', "Kids' Bedding"]
company_struct (object, 193 distinct): ['Trend Lab', 'Sweet JoJo Designs', 'Babies R Us', 'RoomMates', 'Cotton Tale', 'One Grace Place', 'Marmont Hill', 'Bacati', 'Triboro Quilt Co.', 'Lambs & Ivy']
company_free (object, 180 distinct, 78.0% missing): ['Trend Lab', 'Sweet Jojo Designs', 'JoJo Designs', 'Lolli Living', 'aden', 'Majestic Home Goods', 'RoomMates! Simply', 'Northwest', 'Sadie & Scout', 'Pem America']
weight (object, 14 distinct, 99.4% missing): [' 1.5 lbs', ' 0.5 lbs', ' 1 lb. 5 oz.', ' 1 lb. 4 oz.', ' 2 lbs', ' 9 oz', ' 8.6 oz', ' 3 lbs', ' 9.4 oz', ' 4 lbs']
length (object, 96 distinct, 87.6% missing): ['52"', '40"', 'Trend', '39"', '32"', '44"', '2)', '13"', '28"', 'in.']
width (object, 92 distinct, 88.4% missing): ['28"', '30"', '16"', '13"', '52"', '27"', '17.25"', '44"', '12"', 'includes:']
height (object, 52 distinct, 97.4% missing): ['BIGGS', '9"', 'in', '12"', '27"', 'inches', 'the', '9.5"', '14"', '4.75"']
fabrics (object, 48 distinct, 53.2% missing): ['cotton', 'polyester', 'cotton / polyester', 'plush / polyester', 'plush', 'cotton / muslin', 'plush / cotton', 'satin', 'microfiber / polyester', 'microfiber']
colors (object, 146 distinct, 55.3% missing): ['pink', 'blue', 'green', 'gray', 'black', 'grey', 'chocolate', 'purple', 'red', 'green / pink']
materials (object, 14 distinct, 91.9% missing): ['fleece', 'wood', 'microfiber', 'plastic', 'metal', 'wood / pine', 'phthalate', 'metal / plastic', 'polyurethane', 'velcro']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import UrlDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_BABIES_PRICES"
SLUG_BASE = "multabench-babies-prices"
KAGGLE_SOURCE = "http://pages.cs.wisc.edu/~anhai/data/784_data/baby_products/csv_files/babies_r_us.csv"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(UrlDatasetID.REG_TEXT_CONSUMER_BABIES_R_US_PRICES)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
