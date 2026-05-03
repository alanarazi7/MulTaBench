import os
import pandas as pd
from dataclasses import dataclass
from os.path import join
from typing import Optional

from datasets import DatasetDict
from pandas import DataFrame, Series
from pandas.core.dtypes.common import is_numeric_dtype

from multabench.datasets.all_datasets import MultimodalDatasetID
from tabstar.preprocessing.dates import series_to_dt
from tabstar.preprocessing.feat_types import convert_series_to_numeric, convert_series_to_textual
from tabstar.preprocessing.texts import normalize_col_name
from multabench.datasets.curation_mapping import get_curated
from multabench.datasets.curation_objects import CuratedDataset, CuratedTarget, CuratedFeature
from multabench.datasets.multimodal import MultimodalState, filter_by_multimodality
from multabench.datasets.objects import SupervisedTask, FeatureType
from multabench.preprocessing.discretize import discretize_numerical

_DISCRETE_SUFFIX = "_discrete"
MAX_ABS_Z = 5.0

@dataclass
class MultimodalDataset:
    x: DataFrame
    y: Series
    task_type: SupervisedTask
    dataset_id: MultimodalDatasetID
    image_folder: str | None = None

    @property
    def is_cls(self) -> bool:
        return self.task_type != SupervisedTask.REGRESSION


def curate_dataset(x: DataFrame | DatasetDict | None, y: Series | None,
                   dataset_id: MultimodalDatasetID, dir_path: str | None = None,
                   multimodal_state: MultimodalState | None = None,
                   target_override: str | None = None) -> MultimodalDataset:
    curation = get_curated(dataset_id)
    if x is None:
        x = curation.loading_func(dir_path=dir_path)
    elif isinstance(x, DatasetDict):
        x = curation.loading_func(hf_dataset=x)
    if curation.processing_func is not None:
        x = curation.processing_func(df=x)
    x = drop_columns(x, curation=curation)
    if target_override is not None:
        x, y, task_type = _override_target(x=x, target_col=target_override)
    else:
        x, y = set_target_if_missing(x=x, y=y, curation=curation)
        task_type = curation.target.task_type
        y = curate_target_values(y=y, target=curation.target, task_type=task_type)
        y.name = curation.target.new_name
    x = curate_feature_values(x=x, curation=curation)
    x = curate_column_names(x=x, curation=curation)
    x, y = remove_missing_target_rows(x=x, y=y)
    image_folder = _get_image_folder(curation=curation, dir_path=dir_path)
    x, y = remove_missing_image_rows(x=x, y=y, curation=curation, image_folder=image_folder)
    x = filter_by_multimodality(x=x, multimodal_state=multimodal_state, curation=curation)
    validate_task_type(task_type=task_type, y=y)
    dataset = MultimodalDataset(x=x, y=y, task_type=task_type, dataset_id=dataset_id, image_folder=image_folder)
    if task_type == SupervisedTask.REGRESSION:
        check_extreme_outliers(y=y)
    return dataset


def _infer_task(y: Series) -> SupervisedTask:
    if y.nunique() == 2:
        return SupervisedTask.BINARY
    if is_numeric_dtype(y):
        return SupervisedTask.REGRESSION
    return SupervisedTask.MULTICLASS


def _override_target(x: DataFrame, target_col: str) -> tuple[DataFrame, Series, SupervisedTask]:
    """Extract an override target column from x. If target_col ends with '_discrete',
    strip the suffix, find the raw column, discretize it to bins → MULTICLASS.
    Otherwise use the column as-is and infer the task type automatically."""
    if target_col.endswith('_discretize'):
        target_col = target_col.replace('_discretize', _DISCRETE_SUFFIX)
    discretize = target_col.endswith(_DISCRETE_SUFFIX)
    raw_col = target_col[:-len(_DISCRETE_SUFFIX)] if discretize else target_col
    if raw_col not in x.columns:
        raise ValueError(f"Override target '{raw_col}' not found in columns: {list(x.columns)}")
    y = x[raw_col].copy()
    x = x.drop(columns=[raw_col])
    if discretize:
        y = discretize_numerical(y)
        task_type = SupervisedTask.MULTICLASS
    else:
        task_type = _infer_task(y)
    y.name = raw_col
    print(f"[target_override] Using '{raw_col}' as target, task={task_type.name}, discretize={discretize}")
    return x, y, task_type


def drop_columns(x: DataFrame, curation: CuratedDataset) -> DataFrame:
    if curation.cols_to_drop:
        x = x.drop(columns=curation.cols_to_drop, errors='ignore')
    return x

def set_target_if_missing(x: DataFrame, y: Series | None, curation: CuratedDataset) -> tuple[DataFrame, Series]:
    if y is not None:
        return x, y
    target_var = curation.target.raw_name
    if target_var not in x.columns:
        raise ValueError(f"Target variable {target_var} not found! Have: {x.columns}")
    y = x[target_var].copy()
    x = x.drop(columns=[target_var])
    return x, y

def curate_target_values(y: Series, target: CuratedTarget, task_type: SupervisedTask) -> Series:
    if target.processing_func:
        y = y.apply(target.processing_func)
    if target.numeric_missing:
        assert target.task_type == SupervisedTask.REGRESSION
        y = convert_series_to_numeric(y, missing_value=target.numeric_missing)
    if task_type == SupervisedTask.REGRESSION:
        if not is_numeric_dtype(y):
            y = y.astype(float)
        return y
    mapper = target.label_mapping
    if not mapper:
        return y
    y = y.apply(lambda v: target.label_mapping.get(str(v), str(v)))
    return y

def curate_feature_values(x: DataFrame, curation: CuratedDataset) -> DataFrame:
    # TODO: this feels a bit repetitive, can we import other functions?
    for feat in curation.features:
        col = feat.raw_name
        if col not in x.columns:
            continue
        if feat.processing_func is not None:
            x[col] = x[col].apply(feat.processing_func)
        if feat.value_mapping:
            x[col] = x[col].apply(lambda v: feat.value_mapping.get(str(v), str(v)))
        feat_type = get_curated_feat_type(feat)
        if feat_type is None:
            continue
        if feat_type == FeatureType.NUMERIC:
            missing_value = feat.numeric_missing if feat.numeric_missing else None
            x[col] = convert_series_to_numeric(s=x[col], missing_value=missing_value)
        elif feat_type in {FeatureType.CATEGORICAL, FeatureType.TEXT, FeatureType.BOOLEAN}:
            x[col] = convert_series_to_textual(s=x[col])
        elif feat_type == FeatureType.DATE:
            x[col] = series_to_dt(x[col])
        elif feat_type == FeatureType.IMAGE:
            pass
        else:
            raise ValueError(f"Unsupported feature type: {feat_type}")
    return x

def get_curated_feat_type(feat: CuratedFeature) -> Optional[FeatureType]:
    if feat.feat_type is not None:
        return feat.feat_type
    if feat.value_mapping:
        # # The user can be minimalist and provide a mapping, this automatically turns into a non-numeric feature
        return FeatureType.CATEGORICAL
    if feat.numeric_missing is not None:
        return FeatureType.NUMERIC
    return None

def curate_column_names(x: DataFrame, curation: CuratedDataset) -> DataFrame:
    old2new = curation.name_mapper.copy()
    for col in x.columns:
        if col not in old2new:
            old2new[col] = normalize_col_name(col)
    x = x.rename(columns=old2new)
    return x

def remove_missing_target_rows(x: DataFrame, y: Series):
    missing = y.isnull()
    x = x[~missing]
    y = y[~missing]
    return x, y


def _is_missing_image(v, image_folder: str) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return not os.path.exists(os.path.join(image_folder, str(v)))


def remove_missing_image_rows(x: DataFrame, y: Series, curation: CuratedDataset, image_folder: str | None) -> tuple[DataFrame, Series]:
    image_cols = [f.raw_name for f in curation.features if f.feat_type == FeatureType.IMAGE]
    if not image_cols or image_folder is None:
        return x, y
    img_col = image_cols[0]
    # after column renaming, find the new name
    img_col_in_x = curation.name_mapper.get(img_col, img_col)
    if img_col_in_x not in x.columns:
        return x, y
    mask = x[img_col_in_x].apply(lambda v: _is_missing_image(v, image_folder))
    n_missing = mask.sum()
    if n_missing > 0:
        print(f"🗑️ Dropping {n_missing} rows with missing images in column '{img_col_in_x}'")
        x = x[~mask].reset_index(drop=True)
        y = y[~mask].reset_index(drop=True)
    return x, y

def _get_image_folder(curation: CuratedDataset, dir_path: str) -> Optional[str]:
    if curation.image_folder is None:
        return None
    if dir_path is None:
        return curation.image_folder
    return join(dir_path, curation.image_folder)


def validate_task_type(task_type: SupervisedTask, y: Series):
    if task_type == SupervisedTask.REGRESSION:
        if not is_numeric_dtype(y):
            raise ValueError(f"Target variable {y.name} is not numeric")
    elif task_type == SupervisedTask.BINARY:
        if not y.nunique() == 2:
            raise ValueError(f"Target variable {y.name} is not binary")
    # elif task_type == SupervisedTask.MULTICLASS:
    #     if y.nunique() > 10:
    #         raise ValueError(f"Target variable {y.name} has more than 10 classes")



def check_extreme_outliers(y: Series, max_abs_s: float = MAX_ABS_Z) -> Series:
    y_avg = y.mean()
    y_std = y.std()
    y_min = y_avg - max_abs_s * y_std
    y_max = y_avg + max_abs_s * y_std
    if (y < y_min).any() or (y > y_max).any():
        print(f"🚨 Warning: Regression variable {y.name} had extreme outliers!")
        z_scores = (y - y_avg) / y_std
        print(f"  Top 10 z-scores: {[round(z, 3) for z in z_scores.nlargest(10)]}")
        print(f"  Bot 10 z-scores: {[round(z, 3) for z in z_scores.nsmallest(10)]}")
    else:
        print(f"🎯 Regression variable {y.name} had no extreme outliers.")
    # y = y.clip(lower=y_min, upper=y_max)
    return y