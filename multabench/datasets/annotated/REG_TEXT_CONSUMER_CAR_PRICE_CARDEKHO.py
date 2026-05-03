from typing import Any, Optional

import numpy as np
from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedFeature, CuratedTarget
from multabench.datasets.objects import SupervisedTask
from multabench.datasets.utils import load_csv

'''
Dataset Name: sukritchatterjee/used-cars-dataset-cardekho/cars_details_merges.csv
====
Examples: 37814
====
URL: https://www.kaggle.com/sukritchatterjee/used-cars-dataset-cardekho/cars_details_merges.csv
====
Description:
Used Cars Dataset (CarDekho)
A dataset of used cars with all of their details and listing price.

About Dataset
This dataset contains information about ~38000 cars listed on Cardekho.

There are three CSV files in this dataset -

cars_overview.csv : Overview of the cars, contains basic info about the cars such as transmission type, location and the listing price.
car_details.csv : This file contains the almost all the cars in the overview file along with many other details, such as the features of the cars, the type of owner, etc.
car_details_merges.csv : This file is the merged version of the above two files, contains the basic as well as the detailed information of all the cars.
feature_dictionary.csv : Since the data is quite big, this file explains what information each column in the dataset has.

Points to note about the data:

The dataset contains columns which can have duplicate information since the data is scrapped using an API. It is advised to clean the data before using it.
There are multiple unique identifiers for each car, but using usedCarSkuId is recommended.
The price of the cars is under the column named price which has values such as ₹ 3.50 Lakh. We also have another column indicating the price in a continuous variable type called pu
Disclaimer
This data is meant for academic and research purposes and should not be used for commercial purposes.

====
Target Variable: pu (object, 6865 distinct): ['300000', '350000', '400000', '500000', '250000', '450000', '600000', '200000', '550000', '650000']
====
Features:

position (int64, 20 distinct): ['19', '20', '10', '14', '18', '11', '12', '16', '8', '15']
loc (object, 511 distinct): ['Pune City', 'Gurgaon', 'Bangalore City', 'New Delhi G.P.O.', 'pune city', 'gurgaon', 'new delhi g.p.o.', 'bangalore city', 'Mahadevapura', 'Noida']
myear (int64, 34 distinct): ['2017', '2018', '2014', '2015', '2016', '2019', '2013', '2021', '2020', '2012']
bt (object, 11 distinct): ['Hatchback', 'Sedan', 'SUV', 'MUV', 'Minivans', 'Luxury Vehicles', 'Pickup Trucks', 'Convertibles', 'Coupe', 'Wagon']
tt (object, 2 distinct): ['Manual', 'Automatic']
ft (object, 5 distinct): ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
km (object, 23863 distinct): ['70,000', '1,20,000', '80,000', '60,000', '90,000', '50,000', '40,000', '1,10,000', '1,00,000', '35,000']
ip (int64, 2 distinct): ['0', '1']
imgCount (int64, 54 distinct): ['15', '20', '21', '10', '11', '22', '1', '16', '12', '17']
threesixty (bool, 2 distinct): ['0', '1']
dvn (object, 4159 distinct): ['Maruti Swift VXI', 'Maruti Alto 800 LXI', 'Maruti Wagon R LXI CNG', 'Maruti Wagon R VXI BS IV', 'Maruti Swift VDI BSIV', 'Maruti Swift Dzire VDI', 'Honda City 1.5 S MT', 'Hyundai Grand i10 Sportz', 'Maruti Swift Dzire VXI', 'Hyundai i10 Magna']
oem (object, 46 distinct): ['Maruti', 'Hyundai', 'Honda', 'Mahindra', 'Tata', 'Toyota', 'Ford', 'Renault', 'Volkswagen', 'Skoda']
model (object, 382 distinct): ['Honda City', 'Hyundai i20', 'Maruti Swift', 'Maruti Wagon R', 'Maruti Swift Dzire', 'Hyundai i10', 'Hyundai Grand i10', 'Hyundai Creta', 'Hyundai Verna', 'Maruti Baleno']
'''


def remove_commas(text: str) -> str:
    return text.replace(',', '')


def remove_mm_unit(text: Any) -> Optional[float]:
    if isinstance(text, str):
        text = text.strip()
        if '-' in text:
            text = text.split('-')[0]
        for bad_char in ['`', ',']:
            text = text.replace(bad_char, '')
        if text.endswith('mm'):
            text = text[:-2]
        if text.endswith('m'):
            text = text[:-1]
        text = text.strip()
        if not text:
            return None
    if text is np.nan:
        return None
    if text in {'15t2'}:
        return None
    unit = float(text)
    return unit


def load_df(dir_path: str) -> DataFrame:
    return load_csv(dir_path, "cars_details_merges.csv")


TARGET = CuratedTarget(raw_name="pu", task_type=SupervisedTask.REGRESSION, processing_func=remove_commas)
COLS_TO_DROP = ['price', 'price_segment_new', 'price_segment', 'vlink', "price_range_segment", "pi",
                # IDs
                "usedCarSkuId", "ucid", "sid", "dealer_id_new", "dealer_id", "used_carid", "dynx_itemid_x", "usedCarId",
                "dynx_itemid_y",
                # Constant columns
                "page_template", "template_Type_new", "experiment", "dynx_event", "dynx_pagetype", "vehicle_type_new",
                "page_type", "leadFormCta", "offers", "compare", "brandingIcon", "model_type_new", "car_type_new",
                "compare_car_details",
                # Image non-working URLs
                "images",]

FEATURES = [CuratedFeature(raw_name=mmf, new_name=f"{mmf} (mm)", processing_func=remove_mm_unit)
               for mmf in ["Wheel Base", "Front Tread", "Rear Tread", "Length", "Width", "Height",
                           "Ground Clearance Unladen"]]
LOADING_FUNC = load_df
