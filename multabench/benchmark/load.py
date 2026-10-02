"""
Load a MulTaBench dataset from the Hugging Face Hub: a typed data.parquet, metadata.json and, for image datasets,
images-*.zip shards.

Usage:
    from multabench import load_split
    split = load_split("MUL_IMAGE_PETFINDER", fold=0, size="10k")
    split.x_train, split.y_train, split.x_test, split.y_test
"""
import json
from os.path import join

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from multabench.benchmark.splits import SIZE_10K, get_split
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.hub import DATA_PARQUET, extract_images, hf_repo_id
from multabench.datasets.objects import DatasetSplit, MultimodalDataset, SupervisedTask
from multabench.benchmark.utils.constants import METADATA_JSON
from multabench.benchmark.utils.curation import TASK_REG, task_type_from_name


def _parse_task_type(meta: dict, dataset_id, y: pd.Series) -> SupervisedTask:
    task_str = meta.get("task_type") or task_type_from_name(dataset_id.name)
    if task_str == TASK_REG:
        return SupervisedTask.REGRESSION
    return SupervisedTask.BINARY if y.nunique() == 2 else SupervisedTask.MULTICLASS


def _missing_as_nan(df: pd.DataFrame) -> pd.DataFrame:
    # Parquet returns missing strings as None, the CSV reader NaN; TabSTAR counts None as a non-numeric value.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].where(df[col].notna(), np.nan)
    return df


def load_multabench_dataset(dataset_id) -> MultimodalDataset:
    repo_id = hf_repo_id(dataset_id)
    print(f"Downloading {repo_id} from Hugging Face...")
    dir_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
    extract_images(dir_path)
    df = _missing_as_nan(pd.read_parquet(join(dir_path, DATA_PARQUET)))

    with open(join(dir_path, METADATA_JSON)) as f:
        meta = json.load(f)

    target_col = meta["target"]
    image_folder = dir_path  # image paths already include the "images/" prefix

    y = df[target_col]
    x = df.drop(columns=[target_col])

    return MultimodalDataset(x=x, y=y, task_type=_parse_task_type(meta, dataset_id, y), dataset_id=dataset_id, image_folder=image_folder)


def load_split(dataset: str | MulTaBenchDatasetID, fold: int, size: str = SIZE_10K) -> DatasetSplit:
    if isinstance(dataset, str):
        dataset = MulTaBenchDatasetID[dataset]
    return split_dataset(load_multabench_dataset(dataset), fold=fold, size=size)


def split_dataset(dataset: MultimodalDataset, fold: int, size: str = SIZE_10K) -> DatasetSplit:
    train_idx, test_idx = get_split(dataset.y.to_numpy(), is_cls=dataset.is_cls, split=fold, size=size)
    return DatasetSplit(x_train=dataset.x.iloc[train_idx],
                        y_train=dataset.y.iloc[train_idx],
                        x_test=dataset.x.iloc[test_idx],
                        y_test=dataset.y.iloc[test_idx],
                        task_type=dataset.task_type,
                        image_folder=dataset.image_folder)
