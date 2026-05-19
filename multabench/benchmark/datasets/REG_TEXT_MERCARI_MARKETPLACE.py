"""
Dataset Name: REG_TEXT_MERCARI_MARKETPLACE
====
Examples: 100000
====
URL: https://www.openml.org/search?type=data&id=46660
====
Target Variable: log_price (float64, 415 distinct): ['2.3979', '2.5649', '2.7081', '2.8332', '2.3026', '2.1972', '2.7726', '3.0445', '2.0794', '3.2189']
====
Features:

name (object, 94033 distinct): ['Bundle', 'BUNDLE', 'Lularoe TC leggings', 'Reserved', 'Dress', 'Coach purse', 'Vans', 'Converse', 'Nike', 'Miss me jeans']
item_condition_id (int64, 5 distinct): ['1', '3', '2', '4', '5']
category_name (object, 987 distinct, 0.4% missing): ['Women/Athletic Apparel/Pants, Tights, Leggings', 'Women/Tops & Blouses/T-Shirts', 'Beauty/Makeup/Face', 'Beauty/Makeup/Lips', 'Electronics/Video Games & Consoles/Games', 'Beauty/Makeup/Eyes', 'Electronics/Cell Phones & Accessories/Cases, Covers & Skins', 'Women/Underwear/Bras', 'Women/Tops & Blouses/Tank, Cami', 'Women/Tops & Blouses/Blouse']
brand_name (object, 2023 distinct, 42.5% missing): ['Nike', 'PINK', "Victoria's Secret", 'LuLaRoe', 'Apple', 'Lululemon', 'FOREVER 21', 'Nintendo', 'Michael Kors', 'American Eagle']
shipping (int64, 2 distinct): ['0', '1']
item_description (object, 90584 distinct): ['No description yet', 'New', 'Brand new', 'Good condition', 'Great condition', 'Like new', 'Never worn', 'Excellent condition', 'Never used', 'NWT']
cat1 (object, 10 distinct, 0.4% missing): ['Women', 'Beauty', 'Kids', 'Electronics', 'Men', 'Home', 'Vintage & Collectibles', 'Other', 'Handmade', 'Sports & Outdoors']
cat2 (object, 113 distinct, 0.4% missing): ['Athletic Apparel', 'Makeup', 'Tops & Blouses', 'Shoes', 'Jewelry', 'Toys', 'Cell Phones & Accessories', "Women's Handbags", 'Dresses', "Women's Accessories"]
cat3 (object, 705 distinct, 0.4% missing): ['Pants, Tights, Leggings', 'Other', 'Face', 'T-Shirts', 'Shoes', 'Games', 'Lips', 'Athletic', 'Eyes', 'Cases, Covers & Skins']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import OpenMLDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_MERCARI_MARKETPLACE"
SLUG_BASE = "multabench-mercari-marketplace"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46660"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.REG_TEXT_CONSUMER_MERCARI_ONLINE_MARKETPLACE)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
