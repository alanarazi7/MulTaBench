import glob
import os

import pandas as pd

from multabench.leaderboard.elo import RESULTS_DIR, TAR_RESULTS_DIR
from multabench.leaderboard.results import all_results
from multabench.result_keys import ResultKey, RunStatus


def test_all_results_have_one_row_per_model_encoding_dataset_and_fold():
    df = all_results()
    assert not df.duplicated([ResultKey.MODEL, "encoding", ResultKey.DATASET, ResultKey.FOLD]).any()
    assert set(df["encoding"]) == {"frozen", "tar", "end-to-end"}


def test_all_results_have_every_successful_run_in_the_published_csvs():
    paths = glob.glob(os.path.join(RESULTS_DIR, "*.csv")) + glob.glob(os.path.join(TAR_RESULTS_DIR, "*.csv"))
    ok = sum((pd.read_csv(p)[ResultKey.STATUS] == RunStatus.OK).sum() for p in paths)
    assert len(all_results()) == ok
