"""
Dataset Name: REG_IMAGE_DVM_CAR
====
Examples: 100000
====
URL: https://www.kaggle.com/datasets/osamasaifs/dvm-car-with-images
====
Target Variable: Price (float64, 12329 distinct): ['3995.0', '4995.0', '5995.0', '2995.0', '6995.0', '7995.0', '9995.0', '8995.0', '1995.0', '3495.0']
====
Features:

Maker (str, 74 distinct): ['Ford', 'Audi', 'Vauxhall', 'Volkswagen', 'BMW', 'Nissan', 'Peugeot', 'Toyota', 'Citroen', 'Land Rover']
Genmodel (str, 795 distinct): ['Corsa', 'Focus', 'Fiesta', 'Juke', 'X5', 'Astra', 'Mondeo', 'Golf', '500', 'Kuga']
Adv_year (int64, 10 distinct): ['2018', '2021', '2017', '2020', '2016', '2019', '2015', '2014', '2013', '2012']
Adv_month (int64, 13 distinct): ['5', '8', '4', '7', '6', '3', '2', '1', '12', '11']
Color (str, 22 distinct, 8.2% missing): ['Black', 'Silver', 'Blue', 'Grey', 'White', 'Red', 'Green', 'Yellow', 'Brown', 'Orange']
Reg_year (float64, 24 distinct): ['2015.0', '2017.0', '2016.0', '2014.0', '2018.0', '2013.0', '2012.0', '2019.0', '2011.0', '2010.0']
Bodytype (str, 16 distinct, 0.3% missing): ['Hatchback', 'SUV', 'MPV', 'Saloon', 'Coupe', 'Estate', 'Convertible', 'Pickup', 'Combi Van', 'Panel Van']
Runned_Miles (str, 40100 distinct, 0.4% missing): ['10', '100000', '60000', '80000', '70000', '50000', '90000', '75000', '40000', '65000']
Engin_size (str, 66 distinct, 0.7% missing): ['2.0L', '1.6L', '1.2L', '3.0L', '1.4L', '1.0L', '1.5L', '2.2L', '1.8L', '2.5L']
Gearbox (str, 3 distinct, 0.1% missing): ['Manual', 'Automatic', 'Semi-Automatic']
Fuel_type (str, 13 distinct, 0.1% missing): ['Diesel', 'Petrol', 'Hybrid  Petrol/Electric', 'Electric', 'Hybrid  Petrol/Electric Plug-in', 'Petrol Hybrid', 'Petrol Plug-in Hybrid', 'Hybrid  Diesel/Electric', 'Diesel Hybrid', 'Bi Fuel']
Seat_num (float64, 9 distinct, 2.3% missing): ['5.0', '4.0', '7.0', '2.0', '8.0', '6.0', '9.0', '3.0', '1.0']
Door_num (float64, 6 distinct, 1.7% missing): ['5.0', '3.0', '4.0', '2.0', '6.0', '7.0']
Predicted_viewpoint (int64, 9 distinct): ['0', '225', '90', '45', '315', '270', '180', '135', '360']
Quality_check (str, 2 distinct, 74.3% missing): ['P', 'N']
Car Picture (str, 100000 distinct): ['images/Audi_S3_2016_Black_Audi$$S3$$2016$$Black$$7_35$$175$$image_1.jpg', 'images/SEAT_Altea_2011_Blue_SEAT$$Altea$$2011$$Blue$$79_2$$219$$image_0.jpg', 'images/Toyota_Avensis_2009_Blue_Toyota$$Avensis$$2009$$Blue$$92_4$$512$$image_0.jpg', 'images/Toyota_Verso_2014_Silver_Toyota$$Verso$$2014$$Silver$$92_40$$604$$image_1.jpg', 'images/Ford_Edge_2017_Black_Ford$$Edge$$2017$$Black$$29_5$$140$$image_0.jpg', 'images/Jaguar_XE_2016_Black_Jaguar$$XE$$2016$$Black$$39_6$$961$$image_1.jpg', 'images/BMW_3 Series_2001_Green_BMW$$3 Series$$2001$$Green$$8_5$$218$$image_0.jpg', 'images/Volkswagen_up!_2014_White_Volkswagen$$up!$$2014$$White$$95_33$$1106$$image_0.jpg', 'images/Jaguar_XF_2015_Silver_Jaguar$$XF$$2015$$Silver$$39_7$$826$$image_0.jpg', 'images/Citroen_C3_2017_Green_Citroen$$C3$$2017$$Green$$18_8$$25$$image_1.jpg']
"""

import os
from os.path import exists, join
from typing import Any, Optional

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_DVM_CAR"
SLUG_BASE = "multabench-full-dvm-car"
KAGGLE_SOURCE = "osamasaifs/dvm-car-with-images"

TARGET_COL = "Price"
IMAGE_COL = "Car Picture"
_NUM_DIR = "19586296"
IMAGE_SUBFOLDER = join(_NUM_DIR, "resized_DVM_clean", "resized_DVM_clean")
COLS_TO_DROP = ["Adv_ID", "Genmodel_ID", "Image_ID", "Image_name"]

# The source has 246k adverts. Evaluation never uses more than ~12k rows of any dataset, so the
# rest is download weight; capped like Consumer Complaint on the text side.
MAX_ROWS = 100_000
SAMPLE_SEED = 0


def _normalize_price(price: Any) -> Optional[float]:
    if isinstance(price, (int, float)):
        return price
    price = str(price)
    if price.isdigit():
        return float(price)
    if "ukn" in price.lower() or "unk" in price.lower():
        return None
    raise ValueError(f"Cannot normalize price: {price!r}")


def _image_path(image_name: str) -> str:
    """'Bentley$$Arnage$$2000$$Silver$$10_1$$1$$image_0.jpg' lives under Brand/Model/Year/Colour/."""
    folders = "/".join(image_name.split("$$")[:4])
    return join(folders, image_name).replace("'", "")


def _load_and_process(dir_path: str) -> pd.DataFrame:
    tables_dir = join(dir_path, _NUM_DIR, "tables", "tables")
    img_df = pd.read_csv(join(tables_dir, "Image_table.csv"))
    img_df.columns = [c.strip() for c in img_df.columns]
    img_df["Adv_ID"] = img_df["Image_ID"].str.rsplit("$$", n=1).str[0]
    # Several images per advertisement; one is enough and keeps the row/image mapping 1:1.
    img_df = img_df.groupby("Adv_ID").first().reset_index()

    df = pd.read_csv(join(tables_dir, "Ad_table.csv")).merge(img_df, on="Adv_ID", how="inner")
    df.columns = [c.strip() for c in df.columns]
    df[IMAGE_COL] = df["Image_name"].apply(_image_path)
    present = df[IMAGE_COL].apply(lambda p: exists(join(dir_path, IMAGE_SUBFOLDER, p)))
    if not present.all():
        print(f"  Dropping {(~present).sum()} rows whose image file is absent")
    df = df[present]
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df[TARGET_COL] = df[TARGET_COL].apply(_normalize_price)
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    if len(df) > MAX_ROWS:
        print(f"  Sampling {MAX_ROWS} of {len(df)} rows (seed {SAMPLE_SEED})")
        df = df.sample(n=MAX_ROWS, random_state=SAMPLE_SEED).reset_index(drop=True)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER),
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID,
                 slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID} for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
