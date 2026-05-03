from typing import Any, Optional

import numpy as np
from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedFeature, CuratedTarget
from multabench.datasets.objects import SupervisedTask, FeatureType
from multabench.datasets.utils import load_csv

'''
Dataset Name: markusschmitz/museums/museums_prep.csv
====
Examples: 33072
====
URL: https://www.kaggle.com/markusschmitz/museums/museums_prep.csv
====
Description:
Museums
(https://www.kaggle.com/datasets/markusschmitz/museums)
General information on the US museums. The task is to predict the revenues across the museums.

====
Target Variable: Revenue (float64, 10167 distinct): ['0.0', '5840349457.0', '130058221.0', '139978.0', '379772463.0', '3688471185.0', '840046127.0', '938599729.0', '543405959.0', '356118724.0']
====
Features:

Museum Name (object, 21272 distinct): ['WASHINGTON COUNTY HISTORICAL SOCIETY', 'UNION COUNTY HISTORICAL SOCIETY', 'ART GALLERY']
Legal Name (object, 20311 distinct): ['PRESIDENT AND FELLOWS OF HARVARD COLLEGE', 'LAKE COUNTY HISTORICAL SOCIETY']
Alternate Name (object, 1536 distinct): ['SMITHSONIAN INSTITUTION', 'AFRICAN AMERICAN HERITAGE FOUNDATION', 'SLATER MILL HISTORIC SITE']
Museum Type (object, 9 distinct): ['HISTORIC PRESERVATION', 'GENERAL MUSEUM', 'HISTORY MUSEUM', 'ART MUSEUM', 'ARBORETUM, BOTANICAL GARDEN, OR NATURE CENTER']
Institution Name (object, 634 distinct): ['HARVARD UNIVERSITY', 'LUTHER COLLEGE', 'MASSACHUSETTS INSTITUTE OF TECHNOLOGY', 'YALE UNIVERSITY', 'BAYLOR UNIVERSITY']
Street Address (Administrative Location) (object, 16516 distinct): ['603 W JACKSON', 'PO BOX 1', 'PO BOX 25', 'PO BOX 12', 'PO BOX 44']
City (Administrative Location) (object, 7381 distinct): ['NEW YORK', 'CHICAGO', 'WASHINGTON', 'PHILADELPHIA', 'PORTLAND', 'HOUSTON', 'LOS ANGELES']
State (Administrative Location) (object, 51 distinct): ['CA', 'NY', 'TX', 'PA', 'OH', 'IL', 'FL', 'MI', 'MA', 'WI']
Zip Code (Administrative Location) (float64, 12850 distinct): ['74743.0', '92101.0', '2138.0', '19106.0', '4101.0', '70130.0', '10011.0', '19104.0', '10003.0', '17325.0']
City (Physical Location) (object, 3855 distinct): ['NEW YORK', 'WASHINGTON', 'PHILADELPHIA', 'SPRINGFIELD', 'CHICAGO', 'LOS ANGELES']
State (Physical Location) (object, 51 distinct): ['NY', 'CA', 'TX', 'PA', 'OH', 'MA', 'MI', 'VA', 'WI', 'IL']
Phone Number (float64, 1931 distinct): ['2077.0', '6034.0', '4135.0', '8606.0', '2076.0', '2078.0', '3193.0', '8024.0', '8605.0', '6192.0']
Latitude (float64, 19530 distinct): ['42.3698', '43.3146', '42.3599', '39.1872', '32.0842', '40.4416', '41.3096', '45.0497', '31.5463', '39.9579']
Longitude (float64, 19634 distinct): ['-71.1122', '-91.8009', '-71.0943', '-78.1546', '-97.1225', '-81.099', '-122.9784', '-72.9276', '-79.9584', '-87.6243']
Locale Code (NCES) (float64, 4 distinct): ['4.0', '1.0', '2.0', '3.0']
County Code (FIPS) (float64, 286 distinct): ['1.0', '3.0', '31.0', '37.0', '13.0', '5.0', '17.0', '9.0', '61.0', '29.0']
State Code (FIPS) (float64, 51 distinct): ['6.0', '36.0', '42.0', '48.0', '39.0', '17.0', '12.0', '25.0', '26.0', '55.0']
Region Code (AAM) (int64, 6 distinct): ['4', '3', '2', '6', '5', '1']
Tax Period (float64, 62 distinct): ['201312.0', '201412.0', '201406.0', '201306.0', '201409.0', '201405.0', '201212.0', '201403.0', '201404.0', '201309.0']
'''


def phone_number_prefix(phone: Any) -> Optional[int]:
    # Some missing values like '3862550285ext' '6098982300x1' '5852713361x2' '718347FARM' '2124310233x2'
    # Also, some values with decimal points
    if isinstance(phone, float):
        if np.isnan(phone):
            return None
        phone = str(phone)
    phone_prefix = phone[:4]
    return int(phone_prefix)


def load_df(dir_path: str) -> DataFrame:
    return load_csv(dir_path, "museums_prep.csv")


CONTEXT = "General information on the US museums"
TARGET = CuratedTarget(raw_name="Revenue", task_type=SupervisedTask.REGRESSION)
FEATURES = [CuratedFeature(raw_name="Phone Number", processing_func=phone_number_prefix),
            CuratedFeature(raw_name="Zip Code (Administrative Location)", feat_type=FeatureType.NUMERIC)]
COLS_TO_DROP = [
                # ID
                "Unnamed: 0", "Museum ID", "Employer ID Number",
                # Leakage
                "Income"]
LOADING_FUNC = load_df
