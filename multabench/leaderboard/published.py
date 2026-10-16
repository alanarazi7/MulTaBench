"""The leaderboard's runs as published with this package version, one CSV per model in models/."""
import pandas as pd

from multabench.benchmark.collect import RESULT_COLUMNS
from multabench.benchmark.splits import Size
from multabench.leaderboard.elo import load_runs
from multabench.result_keys import ResultKey


def load_published_results(size: Size | str = Size.TEN_K) -> pd.DataFrame:
    """Every published run of that size, failed runs included, in the collect_results columns."""
    runs = load_runs()
    return runs[runs[ResultKey.SIZE] == Size(size)].reindex(columns=RESULT_COLUMNS).reset_index(drop=True)
