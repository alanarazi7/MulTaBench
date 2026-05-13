"""
Load a MulTaBench dataset directly from Kaggle.

Downloads the already-curated dataset (data.csv + images/) using kagglehub
and returns a MultimodalDataset, bypassing the original source and curation logic.

Usage:
    from multabench.datasets.all_datasets import MulTaBenchDatasetID
    from multabench.benchmark.curation.load import load_multabench_dataset
    dataset = load_multabench_dataset(MulTaBenchDatasetID.MUL_IMAGE_LEAGUE_OF_LEGENDS_SKIN_CATEGORY)
"""
import json
import time
from os.path import join
from typing import Optional

import kagglehub
import pandas as pd

from multabench.datasets.curation import MultimodalDataset
from multabench.datasets.multimodal import MultimodalState, MultimodalError
from multabench.datasets.objects import SupervisedTask
from multabench.benchmark.utils.constants import METADATA_JSON, DATA_CSV, MULTABENCH_KAGGLE_OWNER
from multabench.benchmark.utils.curation import TASK_REG, task_type_from_name
from multabench.preprocessing.feat_types import detect_text_features


def _parse_task_type(meta: dict, dataset_id, y: pd.Series) -> SupervisedTask:
    task_str = meta.get("task_type") or task_type_from_name(dataset_id.name)
    if task_str == TASK_REG:
        return SupervisedTask.REGRESSION
    return SupervisedTask.BINARY if y.nunique() == 2 else SupervisedTask.MULTICLASS


def _apply_multimodal_state(x: pd.DataFrame, image_col: Optional[str],
                            multimodal_state: Optional[MultimodalState]) -> pd.DataFrame:
    if multimodal_state is None or multimodal_state == MultimodalState.ALL:
        return x
    if multimodal_state == MultimodalState.IMAGE_ONLY:
        if image_col is None:
            raise MultimodalError("No image column in this MulTaBench dataset")
        return x[[image_col]]
    if multimodal_state == MultimodalState.NON_IMAGE:
        if image_col is None:
            raise MultimodalError("No image column in this MulTaBench dataset")
        x = x.drop(columns=[image_col])
        if len(x.columns) == 0:
            raise MultimodalError("No features left after dropping image column")
        return x
    if multimodal_state == MultimodalState.TEXT_ONLY:
        exclude = {image_col} if image_col else set()
        text_cols = detect_text_features(x, exclude_columns=exclude)
        if not text_cols:
            raise MultimodalError("No text features found in this MulTaBench dataset")
        return x[text_cols]
    if multimodal_state == MultimodalState.NO_TEXT:
        exclude = {image_col} if image_col else set()
        text_cols = detect_text_features(x, exclude_columns=exclude)
        if not text_cols:
            return x
        x = x.drop(columns=text_cols)
        if len(x.columns) == 0:
            raise MultimodalError("No features left after dropping text columns")
        return x
    raise MultimodalError(f"Unsupported multimodal_state for MulTaBench: {multimodal_state}")


def load_multabench_dataset(dataset_id, multimodal_state: Optional[MultimodalState] = None) -> MultimodalDataset:
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
    image_col = meta["image_col"]
    image_folder = dir_path  # image paths in CSV already include "images/" prefix

    y = df[target_col]
    x = df.drop(columns=[target_col])
    x = _apply_multimodal_state(x, image_col, multimodal_state)

    return MultimodalDataset(x=x, y=y, task_type=_parse_task_type(meta, dataset_id, y), dataset_id=dataset_id, image_folder=image_folder)
