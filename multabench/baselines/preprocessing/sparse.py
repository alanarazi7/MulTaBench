"""Sparse columns made dense (copied from tabstar 1.1.15, tabstar/preprocessing/sparse.py)."""
from typing import Optional, Tuple

from pandas import DataFrame, Series, SparseDtype


def densify_objects(x: DataFrame, y: Optional[Series]) -> Tuple[DataFrame, Optional[Series]]:
    for col in x.columns:
        x[col] = densify_series(x[col])
    if y is not None:
        y = densify_series(y)
    return x, y


def densify_series(s: Series) -> Series:
    if not isinstance(s.dtype, SparseDtype):
        return s
    return s.sparse.to_dense()
