"""
Curation script for MUL_IMAGE_PELGAS_BAY_OF_BISCAY_ZOOSCAN_ZOOPLANKTON.

Source: raghavdharwal/pelgas-bay-of-biscay-zooscan-zooplankton-dataset on Kaggle
        1.15M zooplankton individuals imaged with ZooScan from the Bay of Biscay
        (PELGAS surveys, 2004–2016). 127 taxonomic groups.

Task: Filtered to Copepoda suborder (morphologically similar crustaceans, making
      visual discrimination challenging). First 100K rows taken deterministically.
      Target taxon collapsed to top-9 most frequent + "other" (10 classes).

Target: taxon_cat — zooplankton taxonomic category (multiclass, 10 classes).
Features: sample-level measurements (depth, date, station) + object morphological
          descriptors (max feret, convex area, std, skew, etc.) + image.
Dropped: IDs, constants, lineage/taxon columns (direct label leakage), and
         size-related columns that encode the target class implicitly.

Produces:
    <output_dir>/
        data.csv          features + target; image col contains "images/<filename>"
        images/           all images, flat, one file per row
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata

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
