from typing import Any, Optional

from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedFeature, CuratedTarget
from multabench.datasets.objects import SupervisedTask, FeatureType
from multabench.datasets.utils import load_csv

'''
Dataset Name: bogdansorin/second-hand-mercedes-benz-registered-2000-2023-ita/mercedes-benz.csv
====
Examples: 16392
====
URL: https://www.kaggle.com/bogdansorin/second-hand-mercedes-benz-registered-2000-2023-ita/mercedes-benz.csv
====
Description:
Second-hand Mercedes Benz price Italy
This is a dataset created by web scraping second-hand cars italian websites

About Dataset
The dataset include mercedes-benz cars for sale registered from 2000 to 2023
If you found this dataset usefull you can leave a like.

About columns:
brand:manufacturer
model:version of the car
first_reg:year of the first registration of the car
fuel: d for diesel, g for gas, e for electric, l for gpl
mileage_km: mileage of the car in km
seller_type:d for dealer, p for private
shift: manual or automatic
price:the price of the car
power_hp: the power expressed in horse power of the car

====
Target Variable: price (int64, 1737 distinct): ['18000', '29900', '19900', '18500', '26900', '19500', '27900', '25000', '28900', '17500']
====
Features:

model (object, 186 distinct): ['a 180', 'c 220', 'b 180', 'e 220', 'cla 200', 'glc 220', 'gla 200', 'glc 250', 'a 200', 'c 200']
first_reg (datetime64[ns], 0 distinct): ['2019-12-01 00:00:00', '2020-07-01 00:00:00', ...]
fuel (object, 9 distinct): ['d', 'b', '2', '3', 'e', 'c', 'o', 'l', 'unknown']
mileage_km (float64, 6281 distinct): ['150000.0', '130000.0', '120000.0', ...]
seller_type (object, 2 distinct): ['d', 'p']
swift (object, 2 distinct): ['Automatic', 'Manual']
power_hp (float64, 210 distinct): ['136.0', '109.0', '170.0', '116.0', ...]
'''


def process_power_hp(power: Any) -> Optional[int]:
    assert isinstance(power, str)
    digits_power = [c for c in power if c.isdigit()]
    digits_power = ''.join(digits_power)
    if not digits_power:
        return None
    return int(digits_power)


def process_mileage_km(mileage: Any) -> Optional[int]:
    assert isinstance(mileage, str)
    if mileage == 'unknown':
        return None
    return int(mileage)


def load_df(dir_path: str) -> DataFrame:
    return load_csv(dir_path, "mercedes-benz.csv", sep=";")


CONTEXT = "Second-hand cars Mercedes Benz price Italy"
TARGET = CuratedTarget(raw_name="price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = [
    # ID
    'Unnamed: 0',
    # constant
    "brand"]
FEATURES = [CuratedFeature(raw_name="first_reg", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="power_hp", processing_func=process_power_hp),
            CuratedFeature(raw_name="mileage_km", processing_func=process_mileage_km)]
LOADING_FUNC = load_df
