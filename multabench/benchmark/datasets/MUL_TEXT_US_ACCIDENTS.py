"""
Dataset Name: MUL_TEXT_US_ACCIDENTS
====
Examples: 100001
====
URL: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
====
Target Variable: Severity (int64, 4 distinct): ['2', '3', '4', '1']
====
Features:

Source (object, 3 distinct): ['Source1', 'Source2', 'Source3']
Start_Time (object, 89775 distinct, 9.7% missing): ['2021-01-26 16:16:13', '2020-12-16 13:53:25', '2022-06-24 05:06:23', '2021-02-05 00:55:00', '2021-05-03 06:30:28', '2021-04-09 04:32:00', '2021-08-12 15:38:17', '2023-01-27 00:03:00', '2021-07-28 04:06:41', '2022-01-12 08:41:00']
End_Time (object, 90070 distinct, 9.7% missing): ['2021-06-04 17:40:27', '2021-02-22 15:33:40', '2022-06-24 06:34:59', '2018-12-06 09:08:01', '2022-03-10 17:11:17', '2023-03-27 08:35:30', '2016-08-12 18:47:45', '2022-02-09 18:34:13', '2019-08-22 11:01:54', '2023-02-11 01:28:00']
Start_Lat (float64, 90609 distinct): ['40.8451', '42.4613', '40.8448', '44.9661', '26.1881', '42.379', '30.1904', '34.1705', '33.9164', '41.8736']
Start_Lng (float64, 90656 distinct): ['-73.9265', '-83.106', '-93.2699', '-80.3821', '-80.1517', '-73.9255', '-83.217', '-80.7935', '-82.5079', '-97.7705']
End_Lat (float64, 51653 distinct, 43.9% missing): ['25.6843', '25.8894', '25.9415', '25.8911', '25.7332', '28.45', '25.9248', '33.9085', '25.8837', '37.5512']
End_Lng (float64, 51729 distinct, 43.9% missing): ['-80.4166', '-80.1643', '-80.1632', '-80.19', '-80.3358', '-81.4714', '-80.3366', '-80.2933', '-122.6751', '-81.2238']
Distance(mi) (float64, 6181 distinct): ['0.0', '0.01', '0.008', '0.009', '0.01', '0.007', '0.021', '0.025', '0.012', '0.011']
Description (object, 92313 distinct): ['A crash has occurred causing no to minimum delays. Use caution.', 'Accident', 'An unconfirmed report of a crash has been received. Use caution.', 'A crash has occurred use caution.', 'A crash has occurred with minimal delay to traffic. Prepare to slow or move over for worker safety.', 'At I-15 - Accident.', 'Incident on I-95 NB near I-95 Drive with caution.', 'A disabled vehicle is creating a hazard causing no to minimum delays. Use caution.', 'At I-5 - Accident.', 'Incident on I-95 SB near I-95 Drive with caution.']
Street (object, 32294 distinct, 0.1% missing): ['I-95 N', 'I-95 S', 'I-5 N', 'I-10 E', 'I-5 S', 'I-10 W', 'I-80 W', 'I-80 E', 'I-405 N', 'I-75 N']
City (object, 6341 distinct, 0.0% missing): ['Miami', 'Houston', 'Los Angeles', 'Charlotte', 'Dallas', 'Orlando', 'Austin', 'Raleigh', 'Baton Rouge', 'Nashville']
County (object, 1285 distinct): ['Los Angeles', 'Miami-Dade', 'Orange', 'Harris', 'Dallas', 'Mecklenburg', 'Montgomery', 'Wake', 'San Bernardino', 'Maricopa']
State (object, 49 distinct): ['CA', 'FL', 'TX', 'SC', 'NY', 'NC', 'VA', 'PA', 'MN', 'OR']
Zipcode (object, 36542 distinct, 0.0% missing): ['91706', '91761', '92407', '90023', '90703', '92507', '32819', '33186', '92324', '33169']
Country (object, 1 distinct): ['US']
Timezone (object, 4 distinct, 0.1% missing): ['US/Eastern', 'US/Pacific', 'US/Central', 'US/Mountain']
Airport_Code (object, 1615 distinct, 0.3% missing): ['KCQT', 'KRDU', 'KMCJ', 'KCLT', 'KBNA', 'KMIA', 'KORL', 'KBTR', 'KOPF', 'KATT']
Weather_Timestamp (object, 77771 distinct, 1.5% missing): ['2022-03-13 01:53:00', '2022-04-01 16:53:00', '2022-05-02 16:53:00', '2022-05-17 15:53:00', '2022-04-26 16:53:00', '2022-04-11 16:53:00', '2022-05-13 15:53:00', '2021-09-24 15:53:00', '2022-03-13 01:55:00', '2022-01-12 14:53:00']
Temperature(F) (float64, 602 distinct, 2.1% missing): ['77.0', '68.0', '73.0', '72.0', '70.0', '75.0', '64.0', '63.0', '59.0', '66.0']
Wind_Chill(F) (float64, 640 distinct, 25.7% missing): ['73.0', '72.0', '70.0', '77.0', '75.0', '64.0', '68.0', '63.0', '66.0', '79.0']
Humidity(%) (float64, 100 distinct, 2.2% missing): ['93.0', '100.0', '87.0', '90.0', '89.0', '81.0', '96.0', '86.0', '84.0', '94.0']
Pressure(in) (float64, 873 distinct, 1.8% missing): ['29.99', '29.96', '29.94', '29.97', '30.01', '30.03', '29.91', '29.92', '30.0', '30.04']
Visibility(mi) (float64, 47 distinct, 2.3% missing): ['10.0', '7.0', '9.0', '8.0', '5.0', '2.0', '6.0', '4.0', '3.0', '1.0']
Wind_Direction (object, 24 distinct, 2.2% missing): ['CALM', 'S', 'WNW', 'W', 'SSW', 'NW', 'Calm', 'SW', 'WSW', 'SSE']
Wind_Speed(mph) (float64, 71 distinct, 7.4% missing): ['0.0', '5.0', '6.0', '3.0', '7.0', '8.0', '9.0', '10.0', '12.0', '4.6']
Precipitation(in) (float64, 118 distinct, 28.4% missing): ['0.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09']
Weather_Condition (object, 85 distinct, 2.2% missing): ['Fair', 'Mostly Cloudy', 'Cloudy', 'Clear', 'Partly Cloudy', 'Overcast', 'Light Rain', 'Scattered Clouds', 'Light Snow', 'Fog']
Amenity (bool, 2 distinct): ['0', '1']
Bump (bool, 2 distinct): ['0', '1']
Crossing (bool, 2 distinct): ['0', '1']
Give_Way (bool, 2 distinct): ['0', '1']
Junction (bool, 2 distinct): ['0', '1']
No_Exit (bool, 2 distinct): ['0', '1']
Railway (bool, 2 distinct): ['0', '1']
Roundabout (bool, 2 distinct): ['0', '1']
Station (bool, 2 distinct): ['0', '1']
Stop (bool, 2 distinct): ['0', '1']
Traffic_Calming (bool, 2 distinct): ['0', '1']
Traffic_Signal (bool, 2 distinct): ['0', '1']
Turning_Loop (bool, 1 distinct): ['0']
Sunrise_Sunset (object, 2 distinct, 0.3% missing): ['Day', 'Night']
Civil_Twilight (object, 2 distinct, 0.3% missing): ['Day', 'Night']
Nautical_Twilight (object, 2 distinct, 0.3% missing): ['Day', 'Night']
Astronomical_Twilight (object, 2 distinct, 0.3% missing): ['Day', 'Night']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "MUL_TEXT_US_ACCIDENTS"
SLUG_BASE = "multabench-us-accidents"
KAGGLE_SOURCE = "sobhanmoosavi/us-accidents"

SAMPLE_SIZE = 100_000



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.MUL_TEXT_TRANSPORTATION_US_ACCIDENTS_MARCH23)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    if len(df) > SAMPLE_SIZE:
        df = df.groupby(dataset.y.name, group_keys=False).apply(
            lambda g: g.sample(frac=SAMPLE_SIZE / len(df), random_state=42)
        ).reset_index(drop=True)
        print(f"  Sampled to {len(df):,} rows (stratified)")
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
