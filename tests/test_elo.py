import numpy as np
import pandas as pd
import pytest

from multabench.datasets.all_datasets import Modality
from multabench.leaderboard.elo import ANCHOR, INIT, error_table, leaderboard
from multabench.result_keys import STATUS, TEST_ERROR, RunStatus

TEXT = "BIN_TEXT_FAKE_JOB_POSTING"
IMAGE = "MUL_IMAGE_PETFINDER"


def _table(rows):
    return pd.DataFrame(rows, index=[f"d{i}" for i in range(len(rows))])


def test_the_anchor_is_fixed_and_a_better_model_rates_higher():
    rng = np.random.default_rng(0)
    table = _table([{ANCHOR: 0.3, "Good": 0.1 + rng.random() * 0.05, "Bad": 0.5} for _ in range(10)])
    board = leaderboard(table, bootstraps=50).set_index("model")
    assert board.loc[ANCHOR, "elo"] == INIT
    assert board.loc["Good", "elo"] > INIT > board.loc["Bad", "elo"]
    assert board.loc["Good", "elo_low"] <= board.loc["Good", "elo"] <= board.loc["Good", "elo_high"]
    assert board.loc["Good", "best_on"] == 10 and board.loc["Good", "win_rate"] == 1.0


def test_a_missing_result_gives_no_battle_instead_of_being_imputed():
    table = _table([{ANCHOR: 0.3, "Gappy": 0.1, "Bad": 0.5}, {ANCHOR: 0.3, "Gappy": np.nan, "Bad": 0.5},
                    {ANCHOR: 0.3, "Gappy": 0.1, "Bad": 0.5}])
    board = leaderboard(table, bootstraps=10).set_index("model")
    assert board.loc["Gappy", "datasets"] == 2
    assert board.loc["Gappy", "win_rate"] == 1.0


def test_error_table_averages_the_successful_folds_and_filters_modalities():
    runs = pd.DataFrame([
        {"model": ANCHOR, "dataset": TEXT, "fold": 0, STATUS: RunStatus.OK, TEST_ERROR: 0.2},
        {"model": ANCHOR, "dataset": TEXT, "fold": 1, STATUS: RunStatus.OK, TEST_ERROR: 0.4},
        {"model": ANCHOR, "dataset": TEXT, "fold": 2, STATUS: RunStatus.ERROR, TEST_ERROR: np.nan},
        {"model": ANCHOR, "dataset": IMAGE, "fold": 0, STATUS: RunStatus.OK, TEST_ERROR: 1.0},
    ])
    assert error_table(runs).loc[TEXT, ANCHOR] == pytest.approx(0.3)
    assert list(error_table(runs, [Modality.TEXT]).index) == [TEXT]
