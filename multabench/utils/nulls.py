from numbers import Number
from typing import List, Any, Optional, Set

import numpy as np
import pandas as pd
from pandas import Series



def get_invalid_indices(ls: Series) -> Set[int]:
    return {i for i, x in enumerate(ls) if _get_non_null_value(x) is None}


def get_valid_values(ls: Series) -> List:
    return [x for x in ls if _get_non_null_value(x) is not None]

def _get_non_null_value(x: Any) -> Optional[Any]:
    if isinstance(x, str):
        return x
    if isinstance(x, pd.Timestamp):
        if pd.isnull(x):
            return None
    if pd.isna(x):
        return None
    if isinstance(x, Number) and not np.isfinite(x):
        return None
    if pd.isnull(x):
        return None
    return x
