import os

import numpy as np
import pandas as pd
import pytest

from multabench.leaderboard.elo import load_runs
from multabench.leaderboard.export import SPACE_DATA_DIR, cost, leaderboards, per_dataset
from multabench.result_keys import ResultKey


@pytest.fixture(scope="module")
def runs():
    return load_runs()


def test_the_space_leaderboards_follow_the_results(runs):
    shown = pd.read_csv(os.path.join(SPACE_DATA_DIR, "leaderboards.csv"))
    boards = leaderboards(runs, bootstraps=2)
    assert len(boards.groupby(["tier", "task", "modality"])) == 3 * 3 * 7
    for column in ["tier", "task", "modality", ResultKey.MODEL, "datasets"]:
        assert (shown[column] == boards[column]).all()
    assert np.allclose(shown["elo"], boards["elo"], atol=1e-3)


def test_the_space_per_dataset_results_follow_the_results(runs):
    shown = pd.read_csv(os.path.join(SPACE_DATA_DIR, "per_dataset.csv"))
    assert np.allclose(shown[ResultKey.TEST_ERROR], per_dataset(runs)[ResultKey.TEST_ERROR], rtol=1e-5, atol=1e-6)


def test_the_space_cost_follows_the_results(runs):
    shown = pd.read_csv(os.path.join(SPACE_DATA_DIR, "cost.csv")).set_index(ResultKey.MODEL)
    computed = cost(runs).set_index(ResultKey.MODEL).loc[shown.index]
    for column in ["train_s_per_1k", "inference_s_per_1k", "text_encode_s_per_1k", "image_encode_s_per_1k"]:
        assert np.allclose(shown[column], computed[column], atol=1e-3)


def test_raw_text_models_pay_no_text_encoding(runs):
    encode = cost(runs).set_index(ResultKey.MODEL)
    assert (encode.loc[["ConTextTab", "TabSTAR-v1.1"], "text_encode_s_per_1k"] == 0).all()
    assert (encode.drop(["ConTextTab", "TabSTAR-v1.1"])["text_encode_s_per_1k"] > 0).all()
    assert (encode["image_encode_s_per_1k"] > 0).all()


def test_every_model_has_a_developer_logo_and_link(runs):
    models = pd.read_csv(os.path.join(SPACE_DATA_DIR, "models.csv"))
    assert sorted(models[ResultKey.MODEL]) == sorted(runs[ResultKey.MODEL].unique())
    logos = os.path.join(os.path.dirname(SPACE_DATA_DIR), "assets", "logos")
    assert all(os.path.isfile(os.path.join(logos, logo)) for logo in models["logo"])
    assert models["link"].str.startswith("https://").all()
