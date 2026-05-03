"""Semantic feature type classification (text vs categorical). Used for on-the-fly text detection."""

from dataclasses import dataclass, field
from typing import Set

from pandas import DataFrame, Series
from tabstar.preprocessing.feat_types import detect_numerical_features

from multabench.utils.nulls import get_valid_values



MIN_TEXT_UNIQUE_RATIO = 0.8
MIN_TEXT_UNIQUE_FREQUENCY = 100


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


def detect_text_features(
    x: DataFrame, exclude_columns: Set[str] | None = None
) -> list[str]:
    """
    Detect text features on the fly: detect numerical first, then call
    classify_semantic_features and return only the text feature names.
    """
    exclude_columns = set(exclude_columns) if exclude_columns else set()
    x_no_dates = x.select_dtypes(exclude=["datetime", "datetimetz"])
    numerical = set(detect_numerical_features(x_no_dates)) | exclude_columns
    semantic = classify_semantic_features(x=x, numerical_features=numerical)
    return [c for c in semantic.text_features if c not in exclude_columns]
