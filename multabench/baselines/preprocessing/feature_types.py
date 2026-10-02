"""Feature types: image columns, the stored type of every other column, and the dtype conversion per type."""
from dataclasses import dataclass, field
from typing import Any, Set

import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from multabench.baselines.preprocessing.nulls import MISSING_VALUE


def detect_image_features(x: DataFrame) -> list[str]:
    image_columns = []
    for col in x.columns:
        s = x[col].dropna()
        s = s[s != '']
        if is_image_feature(s):
            image_columns.append(col)
    return image_columns

def is_image_feature(series: Series) -> bool:
    if all(is_image_value(value) for value in series):
        return True
    return False

def is_image_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    for suffix in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".jpe", ".gz"]:
        # TODO: ".gz" is not an image format, we'll leave for detection.
        if suffix in value.lower():
            # Not demanding as endswith, since we have "..4927.jpg?width=720&height=720"
            return True
    return False


@dataclass
class FeatureTypes:
    numerical_features: Set[str] = field(default_factory=set)
    categorical_features: Set[str] = field(default_factory=set)
    text_features: Set[str] = field(default_factory=set)


def detect_feature_types(x: DataFrame) -> FeatureTypes:
    result = FeatureTypes()
    for col in x.columns:
        if isinstance(x[col].dtype, pd.CategoricalDtype):
            result.categorical_features.add(col)
        elif is_numeric_dtype(x[col]) or is_bool_dtype(x[col]):
            result.numerical_features.add(col)
        else:
            result.text_features.add(col)
    return result


def transform_feature_types(x: DataFrame, numerical_features: set[str], image_features: set[str]) -> DataFrame:
    new_x = {}
    for col in x.columns:
        if col in numerical_features:
            new_x[col] = x[col].astype(float)
        elif col in image_features:
            new_x[col] = x[col]
        else:
            new_x[col] = x[col].astype(object).fillna(MISSING_VALUE).astype(str)
    new_x = DataFrame(new_x, index=x.index)
    ordered_x = new_x[x.columns]
    return ordered_x
