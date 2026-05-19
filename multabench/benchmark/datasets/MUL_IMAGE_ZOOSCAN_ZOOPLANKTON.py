"""
Dataset Name: MUL_IMAGE_ZOOSCAN_ZOOPLANKTON
====
Examples: 100000
====
URL: https://www.kaggle.com/datasets/raghavdharwal/pelgass-bay-of-biscay-zooscan-zooplankton-dataset
====
Target Variable: taxon_cat (object, 10 distinct): ['Calanoida', 'Oithonidae', 'Calanidae', 'Temoridae', 'Oncaeidae', 'other', 'Acartiidae', 'dead<Copepoda', 'badfocus<Copepoda', 'Centropagidae']
====
Features:

object_lat (float64, 113 distinct): ['46.8653', '44.4692', '47.0825', '47.1936', '46.2047', '46.8178', '46.6558', '45.2267', '46.6633', '46.4311']
object_lon (float64, 118 distinct): ['-4.6947', '-5.0911', '-2.3217', '-5.0183', '-5.3272', '-2.4825', '-4.7617', '-3.5472', '-5.1094', '-2.7794']
object_date (int64, 55 distinct): ['20040520', '20050523', '20040522', '20040502', '20050521', '20060505', '20060509', '20050512', '20040430', '20050506']
object_time (int64, 107 distinct): ['15700', '21400', '222900', '215700', '200200', '25700', '0', '231300', '224500', '11800']
object_depth_max (float64, 83 distinct): ['93.9693', '173.2051', '76.6044', '70.7107', '86.6025', '98.4808', '28.1908', '100.0', '88.3346', '187.9385']
object_stddev (float64, 41573 distinct): ['49.281', '54.71', '51.309', '57.76', '49.034', '60.308', '48.084', '47.674', '46.105', '51.721']
object_mode (int64, 229 distinct): ['243', '242', '241', '240', '239', '238', '237', '236', '254', '72']
object_max (int64, 14 distinct): ['243', '255', '244', '245', '246', '247', '248', '249', '250', '254']
object_circ. (float64, 738 distinct): ['0.138', '0.128', '0.119', '0.13', '0.131', '0.134', '0.147', '0.141', '0.157', '0.144']
object_skew (float64, 2956 distinct): ['0.104', '0.146', '0.053', '0.124', '0.038', '0.117', '0.043', '0.042', '0.132', '0.046']
object_%area (float64, 1721 distinct): ['0.0', '0.04', '0.08', '0.03', '0.09', '0.05', '0.07', '0.06', '0.1', '0.11']
object_fractal (float64, 526 distinct): ['1.187', '1.174', '1.192', '1.182', '1.191', '1.179', '1.181', '1.175', '1.183', '1.185']
object_nb1 (int64, 24 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
object_nb2 (int64, 28 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
object_symetrieh (float64, 8931 distinct): ['3.075', '3.216', '3.076', '3.131', '2.829', '2.914', '3.221', '3.043', '3.532', '3.231']
object_symetriev (float64, 9017 distinct): ['3.216', '3.07', '3.016', '3.591', '3.141', '3.231', '3.159', '3.073', '3.128', '3.153']
object_symetriehc (int64, 22 distinct): ['3', '4', '5', '2', '6', '7', '8', '9', '10', '11']
object_symetrievc (int64, 22 distinct): ['3', '4', '5', '2', '6', '7', '8', '9', '10', '11']
object_fcons (float64, 52429 distinct): ['2.0', '1.653', '0.9', '1.498', '2.557', '1.151', '2.063', '1.615', '2.129', '0.892']
object_thickr (float64, 6485 distinct): ['2.0', '3.0', '2.333', '2.5', '2.667', '2.25', '1.75', '2.4', '2.182', '2.1']
object_meanpos (float64, 83377 distinct): ['-0.9231', '-0.8182', '-0.8519', '-0.7241', '-1.0', '-1.0833', '-1.0408', '-0.7544', '-0.7699', '-1.2222']
object_cv (float64, 73601 distinct): ['31.1526', '37.3134', '36.9004', '28.7356', '44.6429', '30.8642', '40.6504', '28.9855', '36.63', '33.2226']
object_circex (float64, 93019 distinct): ['0.2565', '0.2513', '0.3103', '0.4488', '0.2609', '0.4808', '0.4643', '0.3491', '0.5193', '0.5014']
sample_bottomdepth (float64, 110 distinct): ['105.0', '195.0', '135.0', '139.8', '75.0', '137.9', '354.0', '216.0', '220.8', '225.0']
sample_tot_vol (float64, 47 distinct): ['25.0', '50.0', '7.5', '5.0', '30.0', '46.25', '27.5', '25.5', '5.75', '5.5']
acq_min_mesh (int64, 3 distinct): ['200', '1000', '100']
acq_max_mesh (int64, 4 distinct): ['1000', '999999', '100', '99999']
acq_sub_part (float64, 9 distinct): ['64.0', '32.0', '8.0', '16.0', '4.0', '128.0', '2.0', '1.0', '256.0']
object_image (object, 100000 distinct): ['images/334787176.jpg', 'images/334781782.jpg', 'images/334781783.jpg', 'images/334783769.jpg', 'images/334782870.jpg', 'images/334781784.jpg', 'images/334787580.jpg', 'images/334783770.jpg', 'images/334781984.jpg', 'images/334791074.jpg']
"""

import os
from os.path import join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "MUL_IMAGE_ZOOSCAN_ZOOPLANKTON"
SLUG_BASE = "multabench-zooscan-zooplankton"
KAGGLE_SOURCE = "raghavdharwal/pelgass-bay-of-biscay-zooscan-zooplankton-dataset"

TARGET_COL = "taxon_cat"
IMAGE_COL = "object_image"
TSV_FILE = "101138.tsv"
IMAGE_SUBFOLDER = "101141/individual_images"

SAMPLE_N = 100_000
TOP_K_TAXA = 9


COLS_TO_DROP = [
    "object_id", "objid", "process_id", "acq_id", "sample_stationid", "sample_id", "classif_id",
    "object_depth_min", "process_particle_pixel_size_mm", "sample_net_type", "sample_ship", "sample_program",
    "sample_comment",
    # Direct label leakage
    "object_lineage", "object_taxon",
    # Size features that implicitly encode class
    "object_area_exc", "object_area", "object_esd", "object_perimareaexc",
    "object_intden", "object_major", "object_minor", "object_slope", "object_mean", "object_feret",
    "object_convarea", "object_convperim", "object_centroids", "object_perim", "object_sr",
    "object_range", "object_histcum2", "object_elongation", "object_skelarea", "object_min",
    "object_median", "object_histcum3", "object_cdexc", "object_kurt", "object_histcum1",
    "object_feretareaexc",
]


def _collect_images(df: pd.DataFrame, img_folder: str) -> pd.DataFrame:
    objid2path = {}
    for type_dir in os.listdir(img_folder):
        type_path = join(img_folder, type_dir)
        if not os.path.isdir(type_path):
            continue
        for img_file in os.listdir(type_path):
            if not img_file.endswith(".jpg"):
                continue
            objid = int(img_file.replace(".jpg", ""))
            objid2path[objid] = join(type_dir, img_file)
    df[IMAGE_COL] = df["objid"].map(objid2path)
    return df


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, TSV_FILE), sep="\t")
    df = df[df["object_lineage"].str.contains("Copepoda", na=False)].reset_index(drop=True)
    df = df.iloc[:SAMPLE_N]
    df = _collect_images(df, img_folder=join(dir_path, IMAGE_SUBFOLDER))
    df = df[df[IMAGE_COL].notna()].reset_index(drop=True)
    top = df["object_taxon"].value_counts().nlargest(TOP_K_TAXA).index
    df[TARGET_COL] = df["object_taxon"].where(df["object_taxon"].isin(top), other="other")
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    obj_perim_cols = [c for c in df.columns if c.startswith("object_perim")]
    df = df.drop(columns=obj_perim_cols)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER), dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID, slug=slug,
                 image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
