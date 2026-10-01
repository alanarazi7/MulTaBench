"""Missing values (copied from tabstar 1.1.15, tabstar/preprocessing/nulls.py)."""
from pandas import Series

MISSING_VALUE = "Unknown Value"


def raise_if_null_target(y: Series):
    y_missing = y.isnull().sum()
    if y_missing > 0:
        raise ValueError(f"Target variable {y.name} has {y_missing} null values, please handle them before training.")
