from typing import Any

from pandas import DataFrame, Series
from tabstar.preprocessing.feat_types import convert_series_to_numeric, convert_series_to_textual

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
