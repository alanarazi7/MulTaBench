"""Semantic feature type classification (text vs categorical). Used for on-the-fly text detection."""

from dataclasses import dataclass, field
from typing import Set

import pandas as pd
from pandas import DataFrame, Series
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_numeric_dtype
from tabstar.preprocessing.feat_types import detect_numerical_features

from multabench.utils.nulls import get_valid_values



MIN_TEXT_UNIQUE_RATIO = 0.8
MIN_TEXT_UNIQUE_FREQUENCY = 100
MIN_DATE_PARSE_RATIO = 0.99
_DATE_SAMPLE = 1000
_DATE_SEPARATORS = set("-/:")


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


def _has_date_separator(value: str) -> bool:
    """Guards against decimal-comma numbers, which pandas happily parses as dates."""
    return any(c in _DATE_SEPARATORS or c.isalpha() for c in value)


def is_date_feature(s: Series) -> bool:
    """Dates, whether already typed or still stored as strings, as in the released CSVs."""
    if is_datetime64_any_dtype(s.dtype):
        return True
    if is_numeric_dtype(s.dtype):
        return False
    values = get_valid_values(s)[:_DATE_SAMPLE]
    if not values:
        return False
    if not all(_has_date_separator(str(v)) for v in values):
        return False
    try:
        parsed = pd.to_datetime(pd.Series(values), errors="coerce", format="mixed")
    except Exception:
        return False
    return parsed.notna().mean() >= MIN_DATE_PARSE_RATIO


def detect_text_features(
    x: DataFrame, exclude_columns: Set[str] | None = None
) -> list[str]:
    """
    Detect text features on the fly: detect numerical first, then call
    classify_semantic_features and return only the text feature names.
    Date columns are never text.
    """
    exclude_columns = set(exclude_columns) if exclude_columns else set()
    x_no_dates = x.select_dtypes(exclude=["datetime", "datetimetz"])
    # pandas 3 reads strings as the "str" dtype, which the numerical detector rejects.
    str_cols = [c for c in x_no_dates.columns if x_no_dates[c].dtype == "str"]
    if str_cols:
        x_no_dates = x_no_dates.astype({c: object for c in str_cols})
    numerical = set(detect_numerical_features(x_no_dates)) | exclude_columns
    semantic = classify_semantic_features(x=x_no_dates, numerical_features=numerical)
    return [c for c in semantic.text_features
            if c not in exclude_columns and not is_date_feature(x_no_dates[c])]
