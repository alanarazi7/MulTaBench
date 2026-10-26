import glob
import os

import pandas as pd

from multabench.leaderboard.export import ENCODING_DIRS
from multabench.leaderboard.results import all_results
from multabench.result_keys import ResultKey, RunStatus


def test_all_results_have_one_row_per_model_encoding_dataset_and_fold():
    df = all_results()
    assert not df.duplicated([ResultKey.MODEL, "encoding", ResultKey.DATASET, ResultKey.FOLD]).any()
    assert set(df["encoding"]) == {"frozen", "tar", "large", "end-to-end"}


def test_all_results_have_every_successful_run_in_the_published_csvs():
    paths = [p for d in ENCODING_DIRS.values() for p in glob.glob(os.path.join(d, "*.csv"))]
    ok = sum((pd.read_csv(p)[ResultKey.STATUS] == RunStatus.OK).sum() for p in paths)
    assert len(all_results()) == ok
