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


def _seconds_stored_as_nanoseconds(s: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.to_datetime(s).astype("int64"), unit="s")


DATETIME_COLUMNS: Dict[MulTaBenchDatasetID, Dict[str, Optional[Callable[[pd.Series], pd.Series]]]] = {
    MulTaBenchDatasetID.BIN_TEXT_KICKSTARTER_FUNDING: {"deadline": _seconds_stored_as_nanoseconds,
                                                       "created_at": _seconds_stored_as_nanoseconds},
    MulTaBenchDatasetID.MUL_TEXT_US_ACCIDENTS: {"Start_Time": None, "End_Time": None, "Weather_Timestamp": None},
    MulTaBenchDatasetID.BIN_TEXT_JIGSAW_TOXICITY: {"created_date": None},
    MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES: {"date_first_hired": None},
    MulTaBenchDatasetID.REG_TEXT_VIDEO_GAMES_SALES: {},
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


def dataset_card(dataset_id: MulTaBenchDatasetID, meta: dict, df: pd.DataFrame) -> str:
    columns = "\n".join(f"| {col} | {df[col].dtype} |" for col in df.columns)
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


def upload(dataset_id: MulTaBenchDatasetID, private: bool) -> str:
    if dataset_id not in DATETIME_COLUMNS:
        raise ValueError(f"{dataset_id.name} has no reviewed DATETIME_COLUMNS entry yet")
    kaggle_dir = download_from_kaggle(dataset_id)
    if os.path.isdir(join(kaggle_dir, "images")):
        raise NotImplementedError("Image datasets are not uploaded yet")
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
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=out_dir,
                          commit_message=f"Upload {dataset_id.name}")
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=[d.name for d in MulTaBenchDatasetID])
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    print(f"Uploaded to https://huggingface.co/datasets/{upload(MulTaBenchDatasetID[args.dataset_name], args.private)}")
