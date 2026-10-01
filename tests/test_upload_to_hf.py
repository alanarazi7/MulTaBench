import numpy as np
import pandas as pd
import pytest

from multabench.scripts.upload_to_hf import type_columns


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "target": ["a", "b", "a", "b"],
        "color": ["red", "blue", None, "red"],
        "flag": [True, np.nan, False, True],
        "miles": ["10", "1 mile", "Unknown", np.nan],
        "hired": ["2016-01-02", "2015-05-06", None, "2014-07-08"],
        "price": [1.5, 2.0, np.nan, 3.0],
    })


def test_type_columns():
    typed = type_columns(_df(), target="target", datetime_columns={"hired": None},
                         numeric_columns={"miles": {"1 mile": 1, "Unknown": None}})
    assert typed["target"].dtype == object
    assert isinstance(typed["color"].dtype, pd.CategoricalDtype)
    assert list(typed["flag"].cat.categories) == ["False", "True"]
    assert typed["flag"].isna().sum() == 1
    assert typed["miles"].tolist()[:2] == [10.0, 1.0]
    assert typed["miles"].isna().sum() == 2
    assert pd.api.types.is_datetime64_any_dtype(typed["hired"])
    assert typed["price"].equals(_df()["price"])


def test_type_columns_rejects_an_unmapped_non_number():
    with pytest.raises(ValueError):
        type_columns(_df(), target="target", datetime_columns={}, numeric_columns={"miles": {"1 mile": 1}})
