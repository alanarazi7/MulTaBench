import os

import numpy as np
import pandas as pd
import pytest

from multabench.leaderboard.elo import error_table, leaderboard, load_runs
from multabench.leaderboard.export import DATASET_GROUPS, SPACE_DATA_DIR, per_dataset
from multabench.result_keys import ResultKey


@pytest.fixture(scope="module")
def runs():
    return load_runs()


@pytest.mark.parametrize("group", DATASET_GROUPS)
def test_the_space_leaderboards_follow_the_results(runs, group):
    shown = pd.read_csv(os.path.join(SPACE_DATA_DIR, f"leaderboard_{group}.csv"))
    board = leaderboard(error_table(runs, DATASET_GROUPS[group]), bootstraps=2)
    assert list(shown[ResultKey.MODEL]) == list(board[ResultKey.MODEL])
    assert np.allclose(shown["elo"], board["elo"], atol=1e-3)
    assert (shown["datasets"] == board["datasets"]).all()


def test_the_space_per_dataset_results_follow_the_results(runs):
    shown = pd.read_csv(os.path.join(SPACE_DATA_DIR, "per_dataset.csv"))
    assert np.allclose(shown[ResultKey.TEST_ERROR], per_dataset(runs)[ResultKey.TEST_ERROR], rtol=1e-5, atol=1e-6)
