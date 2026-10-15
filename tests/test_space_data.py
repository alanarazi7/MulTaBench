import os

import numpy as np
import pandas as pd
import pytest

from multabench.leaderboard.elo import load_runs
from multabench.leaderboard.export import SPACE_DATA_DIR, leaderboards, per_dataset
from multabench.result_keys import ResultKey


@pytest.fixture(scope="module")
def runs():
    return load_runs()


def test_the_space_leaderboards_follow_the_results(runs):
    shown = pd.read_csv(os.path.join(SPACE_DATA_DIR, "leaderboards.csv"))
    boards = leaderboards(runs, bootstraps=2)
    assert len(boards.groupby(["task", "modality"])) == 21
    for column in ["task", "modality", ResultKey.MODEL, "datasets"]:
        assert (shown[column] == boards[column]).all()
    assert np.allclose(shown["elo"], boards["elo"], atol=1e-3)


def test_the_space_per_dataset_results_follow_the_results(runs):
    shown = pd.read_csv(os.path.join(SPACE_DATA_DIR, "per_dataset.csv"))
    assert np.allclose(shown[ResultKey.TEST_ERROR], per_dataset(runs)[ResultKey.TEST_ERROR], rtol=1e-5, atol=1e-6)
