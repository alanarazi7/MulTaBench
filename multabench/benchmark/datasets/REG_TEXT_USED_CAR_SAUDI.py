"""
Dataset Name: REG_TEXT_USED_CAR_SAUDI
====
Examples: 8035
====
URL: https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset
====
Target Variable: Price (int64, 541 distinct): ['0', '45000', '35000', '55000', '30000', '50000', '65000', '25000', '40000', '75000']
====
Features:

Make (object, 59 distinct): ['Toyota', 'Hyundai', 'Ford', 'Chevrolet', 'Nissan', 'GMC', 'Kia', 'Lexus', 'Mercedes', 'Mazda']
Type (object, 381 distinct): ['Land Cruiser', 'Camry', 'Hilux', 'Accent', 'Yukon', 'Tahoe', 'Sonata', 'Taurus', 'Elantra', 'Corolla']
Year (int64, 52 distinct): ['2016', '2015', '2017', '2018', '2019', '2014', '2020', '2013', '2012', '2011']
Origin (object, 4 distinct): ['Saudi', 'Gulf Arabic', 'Other', 'Unknown']
Color (object, 15 distinct): ['White', 'Black', 'Silver', 'Grey', 'Another Color', 'Brown', 'Red', 'Golden', 'Blue', 'Navy']
Options (object, 3 distinct): ['Full', 'Standard', 'Semi Full']
Engine_Size (float64, 75 distinct): ['3.5', '2.0', '2.5', '1.6', '2.4', '5.3', '4.0', '2.7', '4.6', '1.5']
Fuel_Type (object, 3 distinct): ['Gas', 'Diesel', 'Hybrid']
Gear_Type (object, 2 distinct): ['Automatic', 'Manual']
Mileage (int64, 2175 distinct): ['300000', '200000', '300', '100000', '90000', '30000', '400000', '180000', '130000', '120000']
Region (object, 27 distinct): ['Riyadh', 'Dammam', 'Jeddah', 'Qassim', 'Al-Medina', 'Al-Ahsa', 'Aseer', 'Makkah', 'Taef', 'Tabouk']
Negotiable (bool, 2 distinct): ['0', '1']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_USED_CAR_SAUDI"
SLUG_BASE = "multabench-full-used-car-saudi"
KAGGLE_SOURCE = "https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_TRANSPORTATION_USED_CAR_SAUDI_ARABIA)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate REG_TEXT_USED_CAR_SAUDI for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
