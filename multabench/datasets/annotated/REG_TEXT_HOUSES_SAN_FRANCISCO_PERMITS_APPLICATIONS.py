from typing import Optional

import pandas as pd
from numpy import isnan
from pandas import DataFrame

from multabench.datasets.utils import load_csv
from multabench.datasets.curation_objects import CuratedTarget
from multabench.datasets.objects import SupervisedTask

LABEL_NAME = "time_to_approve"

'''
Dataset Name: aparnashastry/building-permit-applications-data/
====
Examples: 183796
====
URL: https://www.kaggle.com/aparnashastry/building-permit-applications-data
====
Lennart:
The task is to predict time to approve (in days) for permit applications. Features include structured fields (e.g., permit
type, construction type, estimated cost) and descriptive metadata (e.g., project description, location). While many
permits have zero delay (often legitimate), these are retained. The target's long-tail distribution together with high
frequency of zero-day-permits poses a meaningful challenge for modelling bureaucratic latency with mixed inputs.
====
Description:
About Dataset
Background
A building permit is an official approval document issued by a governmental agency that allows you or your contractor to proceed with a construction or remodeling project on one's property. For more details go to https://www.thespruce.com/what-is-a-building-permit-1398344. Each city or county has its own office related to buildings, that can do multiple functions like issuing permits, inspecting buildings to enforce safety measures, modifying rules to accommodate needs of the growing population etc. For the city of San Francisco, permit issuing is taken care by www.sfdbi.org/

Why is this important: In the recent past, several posts and blogs highlighted that main discrepancy in demand and supply in real estate industry is due to delays in issuing building permits.

Content
The data was downloaded for the dates ranging from Jan 1st, 2013 to Feb 25th, 2018 using the filter in San Francisco open data portal. This is the exact link: https://data.sfgov.org/Housing-and-Buildings/Building-Permits/i98e-djp9/data
There are 43 columns and close to 200k records in the downloaded version (kept here).
====
Target Variable: time_to_approve (float64, 887 distinct): ['0.0', '1.0', '2.0', '3.0', '7.0', '4.0', '6.0', '5.0', '8.0', '14.0']
====
Features:

Permit Type (int64, 8 distinct): ['8', '3', '4', '2', '7', '6', '1', '5']
Permit Type Definition (object, 8 distinct): ['otc alterations permit', 'additions alterations or repairs', 'sign - erect', 'new construction wood frame', 'wall or painted sign', 'demolitions', 'new construction', 'grade or quarry or fill or excavate']
Block (object, 4881 distinct): ['3708', '3735', '7331', '0289', '3709', '3717', '3707', '3721', '3706', '0259']
Lot (object, 1048 distinct): ['001', '007', '002', '003', '006', '008', '009', '005', '004', '011']
Street Number (int64, 5051 distinct): ['1', '101', '100', '50', '201', '555', '2', '55', '350', '150']
Street Number Suffix (object, 18 distinct, 98.9% missing): ['A', 'B', 'V', 'C', 'E', 'F', 'R', 'D', 'K', 'G']
Street Name (object, 1690 distinct): ['Market', 'California', 'Mission', 'Montgomery', 'Geary', '20th', '03rd', 'Folsom', 'Pine', 'Sacramento']
Street Suffix (object, 21 distinct, 1.4% missing): ['St', 'Av', 'Wy', 'Bl', 'Dr', 'Tr', 'Ct', 'Pl', 'Ln', 'Rd']
Unit (float64, 654 distinct, 85.2% missing): ['0.0', '1.0', '2.0', '3.0', '101.0', '4.0', '5.0', '201.0', '6.0', '7.0']
Unit Suffix (object, 150 distinct, 99.0% missing): ['A', 'C', 'B', 'D', 'E', 'HOA', 'F', 'COMML', 'W', 'G']
Description (object, 122498 distinct, 0.0% missing): ['street space', 'reroofing', 'street space permit', 're-roofing', 'streetspace', 'reroofing.', 'street space & sidewalk repair', 'street space w/mta', 'street space and sidewalk repair', 'street space & sidewalk']
Current Status (object, 10 distinct): ['complete', 'issued', 'expired', 'cancelled', 'reinstated', 'suspend', 'revoked', 'approved', 'withdrawn', 'incomplete']
Structural Notification (object, 1 distinct, 97.2% missing): ['Y']
Number of Existing Stories (float64, 64 distinct, 22.1% missing): ['2.0', '3.0', '4.0', '1.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
Number of Proposed Stories (float64, 64 distinct, 22.2% missing): ['2.0', '3.0', '4.0', '1.0', '5.0', '6.0', '7.0', '8.0', '9.0', '11.0']
Voluntary Soft Story Retrofit (object, 1 distinct, 100.0% missing): ['Y']
Fire Only Permit (object, 1 distinct, 90.1% missing): ['Y']
Estimated Cost (float64, 10869 distinct, 20.2% missing): ['1.0', '10000.0', '5000.0', '20000.0', '15000.0', '30000.0', '3000.0', '25000.0', '2000.0', '1000.0']
Revised Cost (float64, 12371 distinct, 0.0% missing): ['1.0', '5000.0', '10000.0', '20000.0', '15000.0', '0.0', '30000.0', '25000.0', '3000.0', '2000.0']
Existing Use (object, 93 distinct, 21.3% missing): ['1 family dwelling', 'apartments', 'office', '2 family dwelling', 'retail sales', 'food/beverage hndlng', 'vacant lot', 'tourist hotel/motel', 'residential hotel', 'warehouse,no frnitur']
Existing Units (float64, 340 distinct, 26.4% missing): ['1.0', '0.0', '2.0', '3.0', '4.0', '6.0', '12.0', '5.0', '8.0', '7.0']
Proposed Use (object, 94 distinct, 22.0% missing): ['1 family dwelling', 'apartments', 'office', '2 family dwelling', 'retail sales', 'food/beverage hndlng', 'tourist hotel/motel', 'residential hotel', 'school', 'warehouse,no frnitur']
Proposed Units (float64, 357 distinct, 26.2% missing): ['1.0', '0.0', '2.0', '3.0', '4.0', '6.0', '12.0', '5.0', '8.0', '7.0']
Plansets (float64, 6 distinct, 20.1% missing): ['2.0', '0.0', '3.0', '4.0', '6.0', '9000.0']
TIDF Compliance (object, 2 distinct, 100.0% missing): ['Y', 'P']
Existing Construction Type (float64, 5 distinct, 22.4% missing): ['5.0', '1.0', '3.0', '2.0', '4.0']
Existing Construction Type Description (object, 5 distinct, 22.4% missing): ['wood frame (5)', 'constr type 1', 'constr type 3', 'constr type 2', 'constr type 4']
Proposed Construction Type (float64, 5 distinct, 22.3% missing): ['5.0', '1.0', '3.0', '2.0', '4.0']
Proposed Construction Type Description (object, 5 distinct, 22.3% missing): ['wood frame (5)', 'constr type 1', 'constr type 3', 'constr type 2', 'constr type 4']
Site Permit (object, 1 distinct, 98.3% missing): ['Y']
Supervisor District (float64, 11 distinct, 0.9% missing): ['3.0', '8.0', '2.0', '6.0', '5.0', '9.0', '7.0', '1.0', '10.0', '4.0']
Neighborhoods   Analysis Boundaries (object, 41 distinct, 0.9% missing): ['Financial District/South Beach', 'Mission', 'Sunset/Parkside', 'West of Twin Peaks', 'Pacific Heights', 'Castro/Upper Market', 'Marina', 'Noe Valley', 'Outer Richmond', 'South of Market']
Zipcode (float64, 27 distinct, 0.9% missing): ['94110.0', '94114.0', '94117.0', '94109.0', '94103.0', '94115.0', '94118.0', '94123.0', '94122.0', '94105.0']
Location_Latitude (float64, 54864 distinct, 0.9% missing): ['37.7923', '37.7929', '37.7286', '37.7752', '37.7934', '37.7898', '37.7906', '37.7771', '37.7767', '37.7841']
Location_Longitude (float64, 54859 distinct, 0.9% missing): ['-122.4035', '-122.3981', '-122.4768', '-122.4175', '-122.3942', '-122.3976', '-122.4017', '-122.4172', '-122.4164', '-122.4061']
'''

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "Building_Permits.csv")
    df = create_time_to_approve(df)
    df = handle_location(df)
    return df


def create_time_to_approve(df: DataFrame):
    filed_date = pd.to_datetime(df['Filed Date'], errors='coerce')
    issued_date = pd.to_datetime(df['Issued Date'], errors='coerce')
    df[LABEL_NAME] = (issued_date - filed_date).dt.days
    df.drop(columns=['Filed Date', 'Issued Date'], inplace=True)
    df = df[df[LABEL_NAME] >= 0]
    df = df[df[LABEL_NAME] <= 1000]
    return df

def handle_location(df: DataFrame):
    df['Location'] = df['Location'].apply(_parse_location)
    df['Location_Latitude'] = df['Location'].apply(_extract_latitude)
    df['Location_Longitude'] = df['Location'].apply(_extract_longitude)
    df.drop(columns=['Location'], inplace=True)
    return df


def _extract_latitude(loc: str) -> Optional[float]:
    if loc is None:
        return None
    lat, lon = loc.split(',')
    return float(lat.strip())

def _extract_longitude(loc: str) -> Optional[float]:
    if loc is None:
        return None
    lat, lon = loc.split(',')
    return float(lon.strip())

def _parse_location(loc: str) -> Optional[str]:
    if isinstance(loc, float) and isnan(loc):
        return None
    return loc.replace('(', '').replace(')', '').strip()


CONTEXT = "San Francisco Houses Building Permits"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = [
    # IDs
    'Permit Number', 'Record ID',
    # Target Leakage
    'Permit Expiration Date', 'Completed Date', 'First Construction Document Date', 'Current Status Date',
    # Duplicate of Filed Date
    'Permit Creation Date',
    # Super-missing
    'Voluntary Soft-Story Retrofit',
]
LOADING_FUNC = load_df
