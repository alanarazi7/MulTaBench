"""Which columns are numerical, and their conversion to float (copied from tabstar 1.1.15,
tabstar/preprocessing/feat_types.py, detection.py and nulls.py)."""
from typing import Any, List, Optional, Set

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype

MAX_NUMERIC_FOR_CATEGORICAL = 50


def detect_numerical_features(x: DataFrame) -> Set[str]:
    return {col for col in x.columns if is_numerical_feature(s=x[col])}


def is_numerical_feature(s: Series) -> bool:
    if is_datetime64_any_dtype(s.dtype):
        raise TypeError(f"At this point, dates should have already been transformed.")
    elif len(_get_valid_values(s)) == 0:
        return False
    elif is_numeric_dtype(s.dtype) or _is_mostly_numerical(s=s):
        return True
    elif is_object_dtype(s.dtype) or is_bool_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return False
    else:
        raise ValueError(f"Unsupported dtype {s.dtype} for series {s.name}")


def convert_series_to_numeric(s: Series, missing_value: Optional[str] = None) -> Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    non_numeric_indices = [not _is_numeric(f) for f in s]
    if not any(non_numeric_indices):
        return s.astype(float)
    if missing_value is None:
        unique_non_numeric = s[non_numeric_indices].unique()
        if len(unique_non_numeric) != 1:
            raise ValueError(f"Missing values detected are {unique_non_numeric}. Should be only one!")
        missing_value = unique_non_numeric[0]
    return s.where(s != missing_value, np.nan).astype(float)


def _is_mostly_numerical(s: Series) -> bool:
    unique = set(_get_valid_values(s))
    if len(unique) <= MAX_NUMERIC_FOR_CATEGORICAL:
        return False
    non_numerical_unique = [v for v in unique if not _is_numeric(v)]
    return len(non_numerical_unique) <= 1


def _is_numeric(f: Any) -> bool:
    if f is None:
        return False
    if isinstance(f, str):
        return f.isdigit()
    if isinstance(f, (int, float,)):
        return True
    try:
        float(f)
        return True
    except ValueError:
        print(f"ValueError: {f} from type {f} cannot be converted to float")
        return False


def _get_valid_values(ls: Series) -> List:
    return [x for x in ls if _get_non_null_value(x) is not None]


def _get_non_null_value(x: Any) -> Optional[Any]:
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return None
    if not np.isfinite(x):
        return None
    if pd.isnull(x):
        return None
    return x
