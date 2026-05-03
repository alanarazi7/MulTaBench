import pandas as pd
from pandas import Series


def discretize_numerical(s: Series, n_bins: int = 10) -> Series:
    """
    Discretize numerical series into quantile-labeled string categories.
    Labels are of the form "Quantile 0-10%", "Quantile 10-20%", etc.
    Coerces to numeric, handles up to n_bins bins (fewer if duplicates exist),
    and returns NaN for non-numeric or missing values.
    """
    s_numeric = pd.to_numeric(s, errors='coerce')
    valid_only = s_numeric.dropna()
    codes, bin_edges = pd.qcut(valid_only, q=n_bins, labels=False, duplicates='drop', retbins=True)
    actual_n = len(bin_edges) - 1
    step = 100 / actual_n
    label_map = {i: f"Quantile {round(i * step)}-{round((i + 1) * step)}%" for i in range(actual_n)}
    binned = codes.map(label_map)
    return binned.reindex(s_numeric.index)