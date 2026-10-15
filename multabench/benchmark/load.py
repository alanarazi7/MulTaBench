"""
Load a MulTaBench dataset from the Hugging Face Hub: a typed data.parquet, metadata.json and, for image datasets,
images-*.zip shards.

Usage:
    from multabench import Size, load_embeddings, load_split
    split = load_split("MUL_IMAGE_PETFINDER", fold=0, size=Size.TEN_K)
    split.x_train, split.y_train, split.x_test, split.y_test
    embeddings = load_embeddings("MUL_TEXT_WINE_REVIEW", fold=0, size=Size.TEN_K, encoder="e5-small")
    embeddings.train["description"], embeddings.test["description"]
    tuned = load_embeddings("MUL_TEXT_WINE_REVIEW", fold=0, size=Size.TEN_K, encoder="e5-small-tar")  # tuned on fold 0
"""
import json
from dataclasses import dataclass
from os.path import join
from typing import Dict, List

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from multabench.benchmark.splits import Size, get_split
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.hub import DATA_PARQUET, extract_images, hf_repo_id
from multabench.datasets.objects import DatasetSplit, MultimodalDataset, SupervisedTask
from multabench.embeddings.hub import (CachedEmbeddings, CachedEncoder, download_embeddings, encoder_inputs,
                                       input_values, is_tuned, parse_encoder, read_embeddings)
from multabench.benchmark.utils.constants import METADATA_JSON
from multabench.benchmark.utils.curation import TASK_REG, task_type_from_name


def _parse_task_type(meta: dict, dataset_id, y: pd.Series) -> SupervisedTask:
    task_str = meta.get("task_type") or task_type_from_name(dataset_id.name)
    if task_str == TASK_REG:
        return SupervisedTask.REGRESSION
    return SupervisedTask.BINARY if y.nunique() == 2 else SupervisedTask.MULTICLASS


def _missing_as_nan(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            # Parquet reads missing strings as None; the CSV reader gave NaN.
            df[col] = df[col].where(df[col].notna(), np.nan)
    return df


def download_dataset(dataset_id: MulTaBenchDatasetID) -> str:
    repo_id = hf_repo_id(dataset_id)
    print(f"Downloading {repo_id} from Hugging Face...")
    dir_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
    extract_images(dir_path)
    return dir_path


def load_multabench_dataset(dataset_id) -> MultimodalDataset:
    dir_path = download_dataset(dataset_id)
    df = _missing_as_nan(pd.read_parquet(join(dir_path, DATA_PARQUET)))

    with open(join(dir_path, METADATA_JSON)) as f:
        meta = json.load(f)

    target_col = meta["target"]
    image_folder = dir_path  # image paths already include the "images/" prefix

    y = df[target_col]
    x = df.drop(columns=[target_col])

    return MultimodalDataset(x=x, y=y, task_type=_parse_task_type(meta, dataset_id, y), dataset_id=dataset_id,
                             image_folder=image_folder, image_column=meta["image_col"])


def load_split(dataset: str | MulTaBenchDatasetID, fold: int, size: Size = Size.TEN_K) -> DatasetSplit:
    if isinstance(dataset, str):
        dataset = MulTaBenchDatasetID[dataset]
    return split_dataset(load_multabench_dataset(dataset), fold=fold, size=size)


def split_dataset(dataset: MultimodalDataset, fold: int, size: Size = Size.TEN_K) -> DatasetSplit:
    train_idx, test_idx = get_split(dataset.y.to_numpy(), is_cls=dataset.is_cls, split=fold, size=size)
    return DatasetSplit(x_train=dataset.x.iloc[train_idx],
                        y_train=dataset.y.iloc[train_idx],
                        x_test=dataset.x.iloc[test_idx],
                        y_test=dataset.y.iloc[test_idx],
                        task_type=dataset.task_type,
                        image_folder=dataset.image_folder,
                        image_column=dataset.image_column)


@dataclass
class SplitEmbeddings:
    """Raw encoder output per feature, (n_rows, dim) and aligned with the split's x_train and x_test rows."""
    train: Dict[str, np.ndarray]
    test: Dict[str, np.ndarray]
    meta: dict


def load_embeddings(dataset: str | MulTaBenchDatasetID, fold: int, size: Size = Size.TEN_K,
                    encoder: str | CachedEncoder = "e5-small") -> SplitEmbeddings:
    if isinstance(dataset, str):
        dataset = MulTaBenchDatasetID[dataset]
    encoder = parse_encoder(encoder)
    y, task_type = load_target(dataset)
    train_idx, test_idx = get_split(y.to_numpy(), is_cls=task_type != SupervisedTask.REGRESSION, split=fold, size=size)
    split = dict(size=size, fold=fold) if is_tuned(encoder) else {}
    cache = read_embeddings(download_embeddings(encoder, dataset, **split))
    x = _load_columns(dataset, columns=cache.meta["features"])
    return SplitEmbeddings(train=_lookup(cache, x.iloc[train_idx], encoder), test=_lookup(cache, x.iloc[test_idx], encoder),
                           meta=cache.meta)


def _lookup(cache: CachedEmbeddings, x: pd.DataFrame, encoder: CachedEncoder) -> Dict[str, np.ndarray]:
    return {col: cache.lookup(encoder_inputs(encoder, col, input_values(x, encoder, col)), feature=col)
            for col in cache.meta["features"]}


def load_target(dataset_id: MulTaBenchDatasetID) -> tuple[pd.Series, SupervisedTask]:
    """The target column alone, without downloading the images."""
    dir_path = _download_table(dataset_id)
    with open(join(dir_path, METADATA_JSON)) as f:
        meta = json.load(f)
    y = _load_columns(dataset_id, columns=[meta["target"]])[meta["target"]]
    return y, _parse_task_type(meta, dataset_id, y)


def _load_columns(dataset_id: MulTaBenchDatasetID, columns: List[str]) -> pd.DataFrame:
    return _missing_as_nan(pd.read_parquet(join(_download_table(dataset_id), DATA_PARQUET), columns=columns))


def _download_table(dataset_id: MulTaBenchDatasetID) -> str:
    return snapshot_download(repo_id=hf_repo_id(dataset_id), repo_type="dataset",
                             allow_patterns=[DATA_PARQUET, METADATA_JSON])
