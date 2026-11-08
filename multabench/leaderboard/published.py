"""The leaderboard's runs as published with this package version, one CSV per model and encoding."""
import pandas as pd

from multabench.benchmark.collect import RESULT_COLUMNS
from multabench.benchmark.splits import DEFAULT_SIZE, Size
from multabench.leaderboard.export import ENCODING, ENTRY, load_entries
from multabench.result_keys import ResultKey


def load_published_results(size: Size | str = DEFAULT_SIZE) -> pd.DataFrame:
    """Every published run of that size, failed runs included, with its encoding and entry, e.g. "LimiX-2 (TILE)"."""
    runs = load_entries()
    runs = runs[runs[ResultKey.SIZE] == Size(size)]
    return runs.reindex(columns=[*RESULT_COLUMNS, ENCODING, ENTRY]).reset_index(drop=True)
