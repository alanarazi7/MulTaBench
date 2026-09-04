"""
Dataset Name: REG_TEXT_RAMEN_RATINGS
====
Examples: 4105
====
URL: https://www.kaggle.com/datasets/ankanhore545/top-ramen-ratings-2022
====
Target Variable: Stars (float64, 39 distinct): ['5.0', '3.5', '3.75', '4.0', '4.5', '3.25', '4.25', '3.0', '2.75', '2.0']
====
Features:

Brand (object, 614 distinct): ['Nissin', 'Maruchan', 'Myojo', 'Nongshim', 'Samyang Foods', 'Paldo', 'Sapporo Ichiban', 'Mama', 'Acecook', 'Indomie']
Variety (object, 3814 distinct): ['Miso Ramen', 'Beef', 'Yakisoba', 'Chicken', 'Artificial Chicken', 'Instant Noodles Chicken Flavour', 'Vegetable', 'Curry Udon', 'Instant Noodles Beef Flavour', 'Tempura Soba']
Style (object, 9 distinct): ['Pack', 'Bowl', 'Cup', 'Tray', 'Box', 'Restaurant', 'Bottle', 'Can', 'Bar']
Country (object, 53 distinct): ['Japan', 'United States', 'South Korea', 'Taiwan', 'China', 'Thailand', 'Malaysia', 'Hong Kong', 'Indonesia', 'Singapore']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_RAMEN_RATINGS"
SLUG_BASE = "multabench-full-ramen-ratings"
KAGGLE_SOURCE = "https://www.kaggle.com/datasets/ankanhore545/top-ramen-ratings-2022"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_FOOD_RAMEN_RATINGS_2022)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate REG_TEXT_RAMEN_RATINGS for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
