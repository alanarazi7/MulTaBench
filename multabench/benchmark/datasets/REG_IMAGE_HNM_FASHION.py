"""
Dataset Name: REG_IMAGE_HNM_FASHION
====
Examples: 104072
====
URL: https://www.kaggle.com/datasets/odins0n/handm-dataset-128x128
====
Target Variable: AvgConsumerAge (float64, 55161 distinct): ['38.0', '36.0', '37.0', '34.0', '35.0', '40.0', '39.0', '41.0', '32.0', '33.0']
====
Features:

prod_name (object, 45345 distinct): ['Dragonfly dress', 'Mike tee', 'Wow printed tee 6.99', '1pk Fun', 'TP Paddington Sweater', 'Pria tee', 'Despacito', 'MY', 'R-NECK SS SLIM FIT', 'DANTE set']
product_type_name (object, 130 distinct): ['Trousers', 'Dress', 'Sweater', 'T-shirt', 'Top', 'Blouse', 'Shorts', 'Jacket', 'Shirt', 'Vest top']
product_group_name (object, 19 distinct): ['Garment Upper body', 'Garment Lower body', 'Garment Full body', 'Accessories', 'Underwear', 'Shoes', 'Swimwear', 'Socks & Tights', 'Nightwear', 'Unknown']
graphical_appearance_name (object, 30 distinct): ['Solid', 'All over pattern', 'Melange', 'Stripe', 'Denim', 'Front print', 'Placement print', 'Check', 'Colour blocking', 'Lace']
colour_group_name (object, 50 distinct): ['Black', 'Dark Blue', 'White', 'Light Pink', 'Grey', 'Light Beige', 'Blue', 'Red', 'Light Blue', 'Greenish Khaki']
perceived_colour_value_name (object, 8 distinct): ['Dark', 'Dusty Light', 'Light', 'Medium Dusty', 'Bright', 'Medium', 'Undefined', 'Unknown']
perceived_colour_master_name (object, 20 distinct): ['Black', 'Blue', 'White', 'Pink', 'Grey', 'Red', 'Beige', 'Green', 'Khaki green', 'Yellow']
department_name (object, 250 distinct): ['Jersey', 'Knitwear', 'Trouser', 'Blouse', 'Swimwear', 'Dress', 'Kids Girl Jersey Fancy', 'Expressive Lingerie', 'Young Girl Jersey Fancy', 'Jersey Fancy']
index_name (object, 10 distinct): ['Ladieswear', 'Divided', 'Menswear', 'Children Sizes 92-140', 'Children Sizes 134-170', 'Baby Sizes 50-98', 'Ladies Accessories', 'Lingeries/Tights', 'Children Accessories, Swimwear', 'Sport']
index_group_name (object, 5 distinct): ['Ladieswear', 'Baby/Children', 'Divided', 'Menswear', 'Sport']
section_name (object, 56 distinct): ['Womens Everyday Collection', 'Divided Collection', 'Baby Essentials & Complements', 'Kids Girl', 'Young Girl', 'Womens Lingerie', 'Girls Underwear & Basics', 'Womens Tailoring', 'Kids Boy', 'Womens Small accessories']
garment_group_name (object, 21 distinct): ['Jersey Fancy', 'Accessories', 'Jersey Basic', 'Knitwear', 'Under-, Nightwear', 'Trousers', 'Blouses', 'Shoes', 'Dresses Ladies', 'Outdoor']
detail_desc (object, 42906 distinct, 0.4% missing): ['T-shirt in printed cotton jersey.', 'Leggings in soft organic cotton jersey with an elasticated waist.', 'T-shirt in soft, printed cotton jersey.', 'Fine-knit trainer socks in a soft cotton blend with elasticated tops.', 'Socks in a soft, jacquard-knit cotton blend with elasticated tops.', 'Socks in a soft, fine-knit cotton blend with elasticated tops.', 'Sunglasses with plastic frames and UV-protective, tinted lenses.', 'Boxer shorts in a cotton weave with an elasticated waist, long legs and button fly.', 'Tights in a soft, fine-knit cotton blend with an elasticated waist.', 'Fine-knit socks in a soft cotton blend.']
ProductPic (object, 104072 distinct): ['images/010_0108775015.jpg', 'images/010_0108775044.jpg', 'images/010_0108775051.jpg', 'images/011_0110065001.jpg', 'images/011_0110065002.jpg', 'images/011_0110065011.jpg', 'images/011_0111565001.jpg', 'images/011_0111565003.jpg', 'images/011_0111586001.jpg', 'images/011_0111593001.jpg']
"""

import os
from os.path import join, exists
from typing import Optional

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_HNM_FASHION"
SLUG_BASE = "multabench-hnm-fashion"
KAGGLE_SOURCE = "odins0n/handm-dataset-128x128"

TARGET_COL = "AvgConsumerAge"
IMAGE_COL = "ProductPic"
IMAGE_SUBFOLDER = "images_128_128"

_COLS_TO_DROP = ["product_code", "product_type_no", "graphical_appearance_no", "colour_group_code",
                 "perceived_colour_value_id", "perceived_colour_master_id", "department_no", "index_code",
                 "index_group_no", "section_no", "garment_group_no"]



def _transform_article_id(article_id: int) -> str:
    return "0" + str(article_id)


def _collect_image(article_id: int, dir_path: str) -> Optional[str]:
    aid = _transform_article_id(article_id)
    prefix = aid[:3]
    path = join(prefix, aid + ".jpg")
    if exists(join(dir_path, IMAGE_SUBFOLDER, path)):
        return path
    return None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "articles.csv"))
    df[IMAGE_COL] = df['article_id'].apply(lambda a: _collect_image(a, dir_path))
    df = df[df[IMAGE_COL].notna()].reset_index(drop=True)

    # Compute average consumer age per article from transactions
    transactions = pd.read_csv(join(dir_path, "transactions_train.csv"))
    transactions['article_id'] = transactions['article_id'].apply(_transform_article_id)
    df['article_id'] = df['article_id'].apply(_transform_article_id)

    customers = pd.read_csv(join(dir_path, "customers.csv"))
    tx_age = transactions[['customer_id', 'article_id']].merge(
        customers[['customer_id', 'age']], on='customer_id')
    article_age = tx_age.groupby('article_id')['age'].mean().reset_index()
    df = df.merge(article_age, on='article_id', how='inner')
    df = df.rename(columns={'age': TARGET_COL})
    df = df.drop(columns=['article_id'])

    drop = [c for c in _COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop)
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER),
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug,
                 image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
