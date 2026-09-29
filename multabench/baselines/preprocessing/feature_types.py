"""Feature type detection: image columns, text vs categorical columns, and the dtype conversion per type."""
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, List, Optional, Set

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from tabstar.preprocessing.feat_types import convert_series_to_numeric, convert_series_to_textual


MIN_TEXT_UNIQUE_RATIO = 0.8
MIN_TEXT_UNIQUE_FREQUENCY = 100


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
class SemanticFeatureTypes:
    categorical_features: Set[str] = field(default_factory=set)
    text_features: Set[str] = field(default_factory=set)


def classify_semantic_features(
    x: DataFrame, numerical_features: Set[str]
) -> SemanticFeatureTypes:
    """Split non-numerical columns into text (high cardinality) vs categorical."""
    result = SemanticFeatureTypes()
    for col in x.columns:
        if col in numerical_features:
            continue
        if _is_text_feature(s=x[col]):
            result.text_features.add(col)
        else:
            result.categorical_features.add(col)
    return result


def _is_text_feature(s: Series) -> bool:
    values = get_valid_values(s)
    if not values:
        return False
    n_unique = len(set(values))
    if n_unique >= MIN_TEXT_UNIQUE_FREQUENCY:
        return True
    unique_ratio = n_unique / len(values)
    return unique_ratio >= MIN_TEXT_UNIQUE_RATIO


def get_valid_values(ls: Series) -> List:
    return [x for x in ls if _get_non_null_value(x) is not None]

def _get_non_null_value(x: Any) -> Optional[Any]:
    if isinstance(x, str):
        return x
    if isinstance(x, pd.Timestamp):
        if pd.isnull(x):
            return None
    if pd.isna(x):
        return None
    if isinstance(x, Number) and not np.isfinite(x):
        return None
    if pd.isnull(x):
        return None
    return x


def transform_feature_types(x: DataFrame, numerical_features: set[str], image_features: set[str]) -> DataFrame:
    new_x = {}
    for col in x.columns:
        if col in numerical_features:
            new_x[col] = convert_series_to_numeric(s=x[col])
        elif col in image_features:
            new_x[col] = x[col]
        else:
            new_x[col] = convert_series_to_textual(s=x[col])
    new_x = DataFrame(new_x, index=x.index)
    ordered_x = new_x[x.columns]
    return ordered_x
