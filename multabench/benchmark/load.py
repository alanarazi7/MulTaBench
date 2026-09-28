"""
Load a MulTaBench dataset directly from Kaggle.

Downloads the already-curated dataset (data.csv + images/) using kagglehub
and returns a MultimodalDataset, bypassing the original source and curation logic.

Usage:
    from multabench.datasets.all_datasets import MulTaBenchDatasetID
    from multabench.benchmark.load import load_multabench_dataset
    dataset = load_multabench_dataset(MulTaBenchDatasetID.MUL_IMAGE_PETFINDER)
"""
import json
import time
from os.path import join

import kagglehub
import pandas as pd

from multabench.datasets.objects import MultimodalDataset, SupervisedTask
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
