"""Feature types, read from the stored dtypes and the dataset's image column, and the dtype conversion per type."""
from dataclasses import dataclass, field
from typing import Optional, Set

import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from multabench.baselines.preprocessing.nulls import MISSING_VALUE


@dataclass
class FeatureTypes:
    numerical_features: Set[str] = field(default_factory=set)
    categorical_features: Set[str] = field(default_factory=set)
    text_features: Set[str] = field(default_factory=set)
    date_features: Set[str] = field(default_factory=set)
    image_features: Set[str] = field(default_factory=set)


def detect_feature_types(x: DataFrame, image_column: Optional[str]) -> FeatureTypes:
    result = FeatureTypes()
    for col in x.columns:
        if col == image_column:
            result.image_features.add(col)
        elif is_datetime64_any_dtype(x[col]):
            result.date_features.add(col)
        elif isinstance(x[col].dtype, pd.CategoricalDtype):
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
