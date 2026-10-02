from typing import Optional

import numpy as np
from pandas import Series
from sklearn.preprocessing import LabelEncoder


def fit_preprocess_y(y: Series, is_cls: bool) -> Optional[LabelEncoder]:
    if not is_cls:
        return None
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    return label_encoder


def transform_preprocess_y(y: Series | np.ndarray, scaler: Optional[LabelEncoder]) -> Series:
    if scaler is None:
        return y
    assert isinstance(scaler, LabelEncoder), f"Scaler must be LabelEncoder if not None, but got {type(scaler)}"
    y = y.copy()
    return Series(scaler.transform(y), name=y.name, index=y.index)
