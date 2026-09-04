"""
Dataset Name: REG_TEXT_USED_CAR_PAKISTAN
====
Examples: 72655
====
URL: https://www.kaggle.com/datasets/mustafaimam/used-car-prices-in-pakistan-2021
====
Target Variable: Price (float64, 2156 distinct): ['750000.0', '650000.0', '1650000.0', '1250000.0', '850000.0', '1350000.0', '1450000.0', '1750000.0', '2250000.0', '2500000.0']
====
Features:

Make (object, 52 distinct): ['Toyota', 'Suzuki', 'Honda', 'Daihatsu', 'KIA', 'Hyundai', 'Nissan', 'Mitsubishi', 'Mercedes', 'FAW']
Model (object, 280 distinct): ['Corolla', 'Civic', 'Mehran', 'Cultus', 'City', 'Alto', 'Vitz', 'Wagon', 'Bolan', 'Vezel']
Version (object, 1328 distinct): ['GLi 1.3 VVTi', 'VXR', 'VX Euro II', 'VX', 'Oriel 1.8 i-VTEC CVT', 'VXR (CNG)', 'XLi VVTi', '1.3 i-VTEC', 'VXR Euro II', 'F 1.0']
Make_Year (int64, 32 distinct): ['2017', '2018', '2015', '2016', '2014', '2019', '2007', '2012', '2013', '2021']
CC (int64, 80 distinct): ['1300', '1000', '800', '1500', '1800', '660', '1600', '2000', '2700', '3000']
Assembly (object, 2 distinct): ['Local', 'Imported']
Mileage (int64, 8132 distinct): ['100000', '150000', '80000', '200000', '70000', '90000', '1', '50000', '60000', '85000']
Registered City (object, 182 distinct): ['Lahore', 'Karachi', 'Islamabad', 'Un-Registered', 'Multan', 'Rawalpindi', 'Faisalabad', 'Peshawar', 'Sialkot', 'Gujranwala']
Transmission (object, 2 distinct): ['Manual', 'Automatic']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_USED_CAR_PAKISTAN"
SLUG_BASE = "multabench-full-used-car-pakistan"
KAGGLE_SOURCE = "https://www.kaggle.com/datasets/mustafaimam/used-car-prices-in-pakistan-2021"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_TRANSPORTATION_USED_CAR_PAKISTAN)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate REG_TEXT_USED_CAR_PAKISTAN for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
