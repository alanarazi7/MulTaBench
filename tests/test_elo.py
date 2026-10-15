import numpy as np
import pandas as pd
import pytest

from multabench.datasets.all_datasets import Modality
from multabench.leaderboard.elo import ANCHOR, INIT, error_table, leaderboard
from multabench.result_keys import ResultKey, RunStatus

TEXT = "BIN_TEXT_FAKE_JOB_POSTING"
IMAGE = "MUL_IMAGE_PETFINDER"


def _table(rows, folds=1):
    """One row per dataset fold: rows[i] is dataset i // folds."""
    index = pd.MultiIndex.from_tuples([(f"d{i // folds}", i % folds) for i in range(len(rows))])
    return pd.DataFrame(rows, index=index)


def test_the_anchor_is_fixed_and_a_better_model_rates_higher():
    rng = np.random.default_rng(0)
    table = _table([{ANCHOR: 0.3, "Good": 0.1 + rng.random() * 0.05, "Bad": 0.5} for _ in range(10)])
    board = leaderboard(table, bootstraps=50).set_index(ResultKey.MODEL)
    assert board.loc[ANCHOR, "elo"] == INIT
    assert board.loc["Good", "elo"] > INIT > board.loc["Bad", "elo"]
    assert board.loc["Good", "elo_low"] <= board.loc["Good", "elo"] <= board.loc["Good", "elo_high"]
    assert board.loc["Good", "best_on"] == 10 and board.loc["Good", "win_rate"] == 1.0


def test_a_missing_result_gives_no_battle_instead_of_being_imputed():
    table = _table([{ANCHOR: 0.3, "Gappy": 0.1, "Bad": 0.5}, {ANCHOR: 0.3, "Gappy": np.nan, "Bad": 0.5},
                    {ANCHOR: 0.3, "Gappy": 0.1, "Bad": 0.5}])
    board = leaderboard(table, bootstraps=10).set_index(ResultKey.MODEL)
    assert board.loc["Gappy", "datasets"] == 2
    assert board.loc["Gappy", "win_rate"] == 1.0


def test_each_fold_is_its_own_battle():
    # Good wins 4 of each dataset's 6 folds by a hair, and loses 2 by a lot, so it has the lower mean error.
    folds = [{ANCHOR: 0.30, "Good": 0.29}] * 4 + [{ANCHOR: 0.30, "Good": 0.31}] * 2
    board = leaderboard(_table(folds * 5, folds=6), bootstraps=20).set_index(ResultKey.MODEL)
    assert board.loc["Good", "win_rate"] == pytest.approx(4 / 6)
    assert board.loc["Good", "best_on"] == 20 and board.loc["Good", "datasets"] == 5
    assert INIT < board.loc["Good", "elo"] < INIT + 400


def test_the_bootstrap_draws_whole_datasets():
    # Every fold of a dataset agrees, so resampling datasets, not folds, keeps the interval wide.
    rows = [{ANCHOR: 0.3, "Coin": 0.1 if i < 2 else 0.5} for i in range(4) for _ in range(6)]
    board = leaderboard(_table(rows, folds=6), bootstraps=200).set_index(ResultKey.MODEL)
    assert board.loc["Coin", "elo_high"] - board.loc["Coin", "elo_low"] > 400


def test_error_table_keeps_each_successful_fold_and_filters_modalities():
    def run(dataset, fold, status, error):
        return {ResultKey.MODEL: ANCHOR, ResultKey.DATASET: dataset, ResultKey.FOLD: fold, ResultKey.STATUS: status,
                ResultKey.TEST_ERROR: error}
    runs = pd.DataFrame([run(TEXT, 0, RunStatus.OK, 0.2), run(TEXT, 1, RunStatus.OK, 0.4),
                         run(TEXT, 2, RunStatus.ERROR, np.nan), run(IMAGE, 0, RunStatus.OK, 1.0)])
    table = error_table(runs)
    assert table[ANCHOR].to_dict() == {(TEXT, 0): 0.2, (TEXT, 1): 0.4, (IMAGE, 0): 1.0}
    assert list(error_table(runs, [Modality.TEXT]).index) == [(TEXT, 0), (TEXT, 1)]
