"""
Dataset Name: REG_TEXT_CHOCOLATE_BAR_RATINGS
====
Examples: 1795
====
URL: https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings
====
Target Variable: Rating (float64, 13 distinct): ['3.5', '3.0', '3.25', '2.75', '3.75', '2.5', '4.0', '2.0', '2.25', '1.5']
====
Features:

Company  (Maker if known) (object, 416 distinct): ['Soma', 'Bonnat', 'Fresco', 'Pralus', 'A. Morin', 'Guittard', 'Domori', 'Arete', 'Valrhona', 'Hotel Chocolat (Coppeneur)']
Specific Bean Origin or Bar Name (object, 1039 distinct): ['Madagascar', 'Peru', 'Ecuador', 'Dominican Republic', 'Venezuela', 'Sambirano', 'Chuao', 'Ocumare', 'Ghana', 'Papua New Guinea']
REF (int64, 440 distinct): ['414', '24', '387', '404', '552', '1450', '32', '1462', '439', '431']
Review Date (int64, 12 distinct): ['2015', '2014', '2016', '2012', '2013', '2011', '2009', '2010', '2008', '2007']
Cocoa Percent (float64, 45 distinct): ['70.0', '75.0', '72.0', '65.0', '80.0', '74.0', '68.0', '60.0', '73.0', '85.0']
Company Location (object, 60 distinct): ['U.S.A.', 'France', 'Canada', 'U.K.', 'Italy', 'Ecuador', 'Australia', 'Belgium', 'Switzerland', 'Germany']
Bean Type (object, 41 distinct, 0.1% missing): ['\xa0', 'Trinitario', 'Criollo', 'Forastero', 'Forastero (Nacional)', 'Blend', 'Criollo, Trinitario', 'Forastero (Arriba)', 'Criollo (Porcelana)', 'Trinitario, Criollo']
Broad Bean Origin (object, 100 distinct, 0.1% missing): ['Venezuela', 'Ecuador', 'Peru', 'Madagascar', 'Dominican Republic', '\xa0', 'Nicaragua', 'Brazil', 'Bolivia', 'Belize']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_CHOCOLATE_BAR_RATINGS"
SLUG_BASE = "multabench-full-chocolate-bar-ratings"
KAGGLE_SOURCE = "https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_FOOD_CHOCOLATE_BAR_RATINGS)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate REG_TEXT_CHOCOLATE_BAR_RATINGS for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
