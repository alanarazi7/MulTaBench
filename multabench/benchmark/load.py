"""
Load a MulTaBench dataset directly from Kaggle.

Downloads the already-curated dataset (data.csv + images/) using kagglehub
and returns a MultimodalDataset, bypassing the original source and curation logic.

Usage:
    from multabench import load_split
    split = load_split("MUL_IMAGE_PETFINDER", fold=0, size="10k")
    split.x_train, split.y_train, split.x_test, split.y_test
"""
import json
import time
from os.path import join

import kagglehub
import pandas as pd

from multabench.benchmark.splits import SIZE_10K, get_split
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.objects import DatasetSplit, MultimodalDataset, SupervisedTask
from multabench.benchmark.utils.constants import METADATA_JSON, DATA_CSV, MULTABENCH_KAGGLE_OWNER
from multabench.benchmark.utils.curation import TASK_REG, task_type_from_name


def _parse_task_type(meta: dict, dataset_id, y: pd.Series) -> SupervisedTask:
    task_str = meta.get("task_type") or task_type_from_name(dataset_id.name)
    if task_str == TASK_REG:
        return SupervisedTask.REGRESSION
    return SupervisedTask.BINARY if y.nunique() == 2 else SupervisedTask.MULTICLASS


def load_multabench_dataset(dataset_id) -> MultimodalDataset:
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

    with open(join(dir_path, METADATA_JSON)) as f:
        meta = json.load(f)

    df = pd.read_csv(join(dir_path, DATA_CSV))

    target_col = meta["target"]
    image_folder = dir_path  # image paths in CSV already include "images/" prefix

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
