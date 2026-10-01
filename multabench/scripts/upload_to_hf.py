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
import time
from os.path import join
from typing import Callable, Dict, Union

import kagglehub
import pandas as pd
from huggingface_hub import CommitOperationDelete, HfApi

from multabench.benchmark.utils.constants import DATA_CSV, METADATA_JSON, MULTABENCH_KAGGLE_OWNER
from multabench.datasets.all_datasets import MULTABENCH_SOURCES, MulTaBenchDatasetID
from multabench.datasets.hub import DATA_PARQUET, IMAGE_SHARDS, hf_repo_id, pack_images

MAX_CATEGORIES = 100


def _seconds_stored_as_nanoseconds(s: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.to_datetime(s).astype("int64"), unit="s")


def _yyyymmdd_integers(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), format="%Y%m%d")


# Reviewed by hand for every dataset; a dataset missing here has no datetime columns. Years stored as numbers and dates
# embedded in longer strings (e.g. "8 September 1960 (USA)") are left as they are. A column maps to None (ISO-like,
# parsed with format="mixed"), to an explicit format for day-first dates, or to a parser for a broken stored form.
DATETIME_COLUMNS: Dict[MulTaBenchDatasetID, Dict[str, Union[None, str, Callable[[pd.Series], pd.Series]]]] = {
    MulTaBenchDatasetID.BIN_TEXT_KICKSTARTER_FUNDING: {"deadline": _seconds_stored_as_nanoseconds,
                                                       "created_at": _seconds_stored_as_nanoseconds},
    MulTaBenchDatasetID.MUL_TEXT_US_ACCIDENTS: {"Start_Time": None, "End_Time": None, "Weather_Timestamp": None},
    MulTaBenchDatasetID.BIN_TEXT_JIGSAW_TOXICITY: {"created_date": None},
    MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES: {"date_first_hired": None},
    MulTaBenchDatasetID.MUL_TEXT_CONSUMER_COMPLAINT: {"Date received": None, "Date sent to company": None},
    MulTaBenchDatasetID.MUL_TEXT_BOX_OFFICE: {"release_date": None},
    MulTaBenchDatasetID.BIN_TEXT_OSHA_INJURY: {"Event Date": None},
    MulTaBenchDatasetID.MUL_TEXT_MELBOURNE_AIRBNB: {"host_since": None, "first_review": None, "last_review": None},
    MulTaBenchDatasetID.BIN_TEXT_CALIFORNIA_PRICES: {"Listed_On": None, "Last_Sold_On": None},
    MulTaBenchDatasetID.MUL_TEXT_BOOKS_GOODREADS: {"PublishDate": None},
    MulTaBenchDatasetID.REG_TEXT_AIRBNB_SEATTLE: {"host_since": None, "first_review": None, "last_review": None},
    MulTaBenchDatasetID.MUL_IMAGE_ZOOSCAN_ZOOPLANKTON: {"object_date": _yyyymmdd_integers},
    MulTaBenchDatasetID.MUL_IMAGE_REDDIT_MEMES: {"Cake Day": "%d-%m-%Y"},
    MulTaBenchDatasetID.REG_IMAGE_KAMERNET_SIZE: {"from_date": "%d %b %Y"},
    MulTaBenchDatasetID.REG_IMAGE_SAO_PAULO_HOUSES: {"listing date": "%d/%m/%Y"},
    MulTaBenchDatasetID.REG_IMAGE_AIRBNB_NYC: {"host_since": None, "first_review": None, "last_review": None},
}


def download_from_kaggle(dataset_id: MulTaBenchDatasetID) -> str:
    slug = dataset_id.value
    kaggle_ref = f"{MULTABENCH_KAGGLE_OWNER}/{slug}"
    print(f"Downloading {kaggle_ref} from Kaggle...")
    for attempt in range(3):
        try:
            dir_path = kagglehub.dataset_download(kaggle_ref)
            break
        except FileNotFoundError as e:
            if attempt == 2:
                raise
            wait = 60 * (attempt + 1) * 5  # 5 min, 10 min
            print(f"kagglehub archive bug (attempt {attempt + 1}/3): {e} — retrying in {wait // 60} min...")
            time.sleep(wait)
    print(f"💾 Downloaded to: {dir_path}")
    return dir_path


def _is_string(s: pd.Series) -> bool:
    return s.dtype == object and s.dropna().map(type).eq(str).all()


def type_columns(df: pd.DataFrame, target: str, datetime_columns: dict) -> pd.DataFrame:
    typed = df.copy()
    for col in df.columns:
        if col == target or col in datetime_columns:
            continue
        if _is_string(df[col]) and df[col].nunique() < MAX_CATEGORIES:
            typed[col] = df[col].astype("category")
    for col, parse in datetime_columns.items():
        if parse is None:
            typed[col] = pd.to_datetime(df[col], format="mixed")
        elif isinstance(parse, str):
            typed[col] = pd.to_datetime(df[col], format=parse)
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
    if _is_string(s):
        return "string"
    value_types = sorted({type(v).__name__ for v in s.dropna()})
    return f"mixed ({', '.join(value_types)})"


def dataset_card(dataset_id: MulTaBenchDatasetID, meta: dict, df: pd.DataFrame) -> str:
    types = {col: _column_type(df[col]) for col in df.columns}
    if meta.get("image_col"):
        types[meta["image_col"]] = f"image path (inside `{IMAGE_SHARDS}`)"
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


def _pack_images(paths: pd.Series, src_dir: str, out_dir: str):
    missing = [p for p in paths.dropna() if not os.path.isfile(join(src_dir, p))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing, e.g. {missing[:3]}")
    print(f"Packed {pack_images(src_dir, out_dir)} images")


def _delete_stale_files(api: HfApi, repo_id: str, out_dir: str):
    local = {os.path.relpath(join(root, f), out_dir) for root, _, names in os.walk(out_dir) for f in names}
    stale = [f for f in api.list_repo_files(repo_id, repo_type="dataset") if f not in local and f != ".gitattributes"]
    if stale:
        api.create_commit(repo_id=repo_id, repo_type="dataset", commit_message=f"Remove {len(stale)} stale files",
                          operations=[CommitOperationDelete(path_in_repo=f) for f in stale])


def upload(dataset_id: MulTaBenchDatasetID) -> str:
    kaggle_dir = download_from_kaggle(dataset_id)
    with open(join(kaggle_dir, METADATA_JSON)) as f:
        meta = json.load(f)
    df = pd.read_csv(join(kaggle_dir, DATA_CSV), low_memory=False)
    typed = type_columns(df, target=meta["target"], datetime_columns=DATETIME_COLUMNS.get(dataset_id, {}))

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
            _pack_images(df[meta["image_col"]], src_dir=kaggle_dir, out_dir=out_dir)
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
        api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=out_dir)
        _delete_stale_files(api, repo_id, out_dir)
    return repo_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=[d.name for d in MulTaBenchDatasetID])
    args = parser.parse_args()
    print(f"Uploaded to https://huggingface.co/datasets/{upload(MulTaBenchDatasetID[args.dataset_name])}")
