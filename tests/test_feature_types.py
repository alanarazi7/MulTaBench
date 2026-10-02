import numpy as np
import pandas as pd

from multabench.baselines.preprocessing.feature_types import detect_feature_types


def test_feature_types_follow_the_stored_dtypes():
    x = pd.DataFrame({
        "price": [1.5, np.nan, 3.0],
        "count": [1, 2, 3],
        "flag": [True, False, True],
        "color": pd.Categorical(["red", "blue", None]),
        "review": ["great", None, "awful"],
        "listed": pd.to_datetime(["2020-01-01", None, "2021-06-30"]),
        "photo": ["images/a.jpg", "images/b.jpg", None],
    })
    types = detect_feature_types(x, image_column="photo")
    assert types.numerical_features == {"price", "count", "flag"}
    assert types.categorical_features == {"color"}
    assert types.text_features == {"review"}
    assert types.date_features == {"listed"}
    assert types.image_features == {"photo"}
