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
    })
    types = detect_feature_types(x)
    assert types.numerical_features == {"price", "count", "flag"}
    assert types.categorical_features == {"color"}
    assert types.text_features == {"review"}
