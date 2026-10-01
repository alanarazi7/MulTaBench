"""
Upload a MulTaBench dataset from Kaggle to the Hugging Face Hub as typed Parquet.

Columns and values are kept as they are. Only two types are added: the datetime columns listed in DATETIME_COLUMNS
are parsed, and string features with fewer than MAX_CATEGORIES distinct values become categorical.

Usage:
    python -m multabench.scripts.upload_to_hf --dataset_name REG_TEXT_MONTGOMERY_SALARIES
"""
import argparse
import json
import os
import shutil
import tempfile
from os.path import join
from typing import Callable, Dict, Optional

import pandas as pd
from huggingface_hub import HfApi

from multabench.benchmark.load import download_from_kaggle
from multabench.benchmark.utils.constants import DATA_CSV, METADATA_JSON
from multabench.datasets.all_datasets import MULTABENCH_SOURCES, MulTaBenchDatasetID
from multabench.datasets.hub import DATA_PARQUET, hf_repo_id

MAX_CATEGORIES = 100
IMAGES_DIR = "images"


def _seconds_stored_as_nanoseconds(s: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.to_datetime(s).astype("int64"), unit="s")


DATETIME_COLUMNS: Dict[MulTaBenchDatasetID, Dict[str, Optional[Callable[[pd.Series], pd.Series]]]] = {
    MulTaBenchDatasetID.BIN_TEXT_KICKSTARTER_FUNDING: {"deadline": _seconds_stored_as_nanoseconds,
                                                       "created_at": _seconds_stored_as_nanoseconds},
    MulTaBenchDatasetID.MUL_TEXT_US_ACCIDENTS: {"Start_Time": None, "End_Time": None, "Weather_Timestamp": None},
    MulTaBenchDatasetID.BIN_TEXT_JIGSAW_TOXICITY: {"created_date": None},
    MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES: {"date_first_hired": None},
    MulTaBenchDatasetID.REG_TEXT_VIDEO_GAMES_SALES: {},
    MulTaBenchDatasetID.REG_IMAGE_KHAADI_CLOTHES: {},
}


def type_columns(df: pd.DataFrame, target: str, datetime_columns: dict) -> pd.DataFrame:
    typed = df.copy()
    for col in df.columns:
        if col == target or col in datetime_columns:
            continue
        if df[col].dtype == object and df[col].nunique() < MAX_CATEGORIES:
            typed[col] = df[col].astype("category")
    for col, parse in datetime_columns.items():
        if parse is None:
            typed[col] = pd.to_datetime(df[col], format="mixed")
        else:
            typed[col] = parse(df[col])
    for col in df.columns:
        if typed[col].isna().sum() != df[col].isna().sum():
            raise ValueError(f"Typing {col} changed its number of missing values")
    return typed


def _column_type(s: pd.Series) -> str:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    return "string"


def dataset_card(dataset_id: MulTaBenchDatasetID, meta: dict, df: pd.DataFrame) -> str:
    types = {col: _column_type(df[col]) for col in df.columns}
    if meta.get("image_col"):
        types[meta["image_col"]] = f"image path (under `{IMAGES_DIR}/`)"
    columns = "\n".join(f"| {col} | {kind} |" for col, kind in types.items())
    return f"""---
tags:
- multabench
- tabular
---

# {dataset_id.name}

Part of [MulTaBench](https://github.com/alanarazi7/MulTaBench), a benchmark for multimodal tabular learning.

- **Target:** `{meta["target"]}`
- **Rows:** {len(df)}
- **Original source:** {MULTABENCH_SOURCES.get(dataset_id, "unknown")}

Load the benchmark splits with `multabench.load_split("{dataset_id.name}", fold=0)`.

| Column | Type |
|---|---|
{columns}
"""


def _copy_images(paths: pd.Series, src_dir: str, out_dir: str):
    missing = [p for p in paths.dropna() if not os.path.isfile(join(src_dir, p))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing, e.g. {missing[:3]}")
    shutil.copytree(join(src_dir, IMAGES_DIR), join(out_dir, IMAGES_DIR))


def upload(dataset_id: MulTaBenchDatasetID) -> str:
    if dataset_id not in DATETIME_COLUMNS:
        raise ValueError(f"{dataset_id.name} has no reviewed DATETIME_COLUMNS entry yet")
    kaggle_dir = download_from_kaggle(dataset_id)
    with open(join(kaggle_dir, METADATA_JSON)) as f:
        meta = json.load(f)
    df = pd.read_csv(join(kaggle_dir, DATA_CSV), low_memory=False)
    typed = type_columns(df, target=meta["target"], datetime_columns=DATETIME_COLUMNS[dataset_id])

    repo_id = hf_repo_id(dataset_id)
    with tempfile.TemporaryDirectory() as out_dir:
        typed.to_parquet(join(out_dir, DATA_PARQUET), index=False)
        reread = pd.read_parquet(join(out_dir, DATA_PARQUET))
        if reread.shape != df.shape or not reread.dtypes.astype(str).equals(typed.dtypes.astype(str)):
            raise ValueError("The Parquet round trip changed the data")
        with open(join(out_dir, METADATA_JSON), "w") as f:
            json.dump(meta, f, indent=2)
        with open(join(out_dir, "README.md"), "w") as f:
            f.write(dataset_card(dataset_id, meta, typed))
        if meta.get("image_col"):
            _copy_images(df[meta["image_col"]], src_dir=kaggle_dir, out_dir=out_dir)
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=out_dir,
                          commit_message=f"Upload {dataset_id.name}")
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=[d.name for d in MulTaBenchDatasetID])
    args = parser.parse_args()
    print(f"Uploaded to https://huggingface.co/datasets/{upload(MulTaBenchDatasetID[args.dataset_name])}")
