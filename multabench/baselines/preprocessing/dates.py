"""Datetime columns expanded into numeric parts (copied from tabstar 1.1.15, tabstar/preprocessing/dates.py)."""
from typing import Dict, Set

import pandas as pd
from pandas import DataFrame
from skrub import DatetimeEncoder


def transform_date_features(x: DataFrame, date_transformers: Dict[str, DatetimeEncoder]) -> DataFrame:
    rows = x.shape[0]
    for col, dt_encoder in date_transformers.items():
        dt_df = dt_encoder.transform(x[col])
        dt_df.index = x.index
        x = x.drop(columns=[col])
        x = pd.concat([x, dt_df], axis=1)
        if x.shape[0] != rows:
            raise ValueError(f"Row mismatch after transforming date column {col}: expected {rows}, got {x.shape}")
    return x


def fit_date_encoders(x: DataFrame, date_features: Set[str]) -> Dict[str, DatetimeEncoder]:
    date_encoders = {}
    for col in [c for c in x.columns if c in date_features]:
        # Adds: "year", "month", "day", "hour", "total_seconds", "weekday"
        encoder = DatetimeEncoder(add_weekday=True, add_total_seconds=True)
        encoder.fit(x[col])
        date_encoders[col] = encoder
    return date_encoders

