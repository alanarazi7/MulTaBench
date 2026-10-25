import os

import numpy as np
import pandas as pd
import pytest

from multabench.leaderboard.export import (ENCODINGS, ENTRY, SPACE_DATA_DIR, cost, encoding_leaderboards, leaderboards,
                                           load_entries, per_dataset)
from multabench.result_keys import ResultKey


@pytest.fixture(scope="module")
def runs():
    return load_entries()


def test_the_space_leaderboards_follow_the_results(runs):
    shown = pd.read_csv(os.path.join(SPACE_DATA_DIR, "leaderboards.csv"))
    boards = leaderboards(runs, bootstraps=2)
    assert len(boards.groupby(["tier", "task", "modality", "encodings"])) == 3 * 3 * 3 * (2 ** len(ENCODINGS) - 1)
    # Entries whose Elo ties can sort either way from one machine to another, so the boards are compared by entry.
    key = ["tier", "task", "modality", "encodings", ENTRY]
    shown, boards = shown.set_index(key).sort_index(), boards.set_index(key).sort_index()
    assert shown.index.equals(boards.index)
    for column in [ResultKey.MODEL, "encoding", "datasets"]:
        assert (shown[column] == boards[column]).all()
    assert np.allclose(shown["elo"], boards["elo"], atol=1e-3)


def test_the_space_encoding_boards_follow_the_results(runs):
    shown = pd.read_csv(os.path.join(SPACE_DATA_DIR, "encodings.csv"))
    boards = encoding_leaderboards(runs, bootstraps=2)
    assert len(boards.groupby(["tier", "task", "modality"])) == 3 * 3 * 3
    for column in ["tier", "task", "modality", "encoding", "datasets", "models"]:
        assert (shown[column] == boards[column]).all()
    assert np.allclose(shown["elo"], boards["elo"], atol=1e-3)


def test_each_model_reading_embeddings_has_an_entry_per_encoding(runs):
    encodings = runs.groupby(ResultKey.MODEL)["encoding"].unique().map(sorted)
    assert set(encodings[["ConTextTab", "TabSTAR-v1.1"]].map(tuple)) == {("end-to-end",)}
    assert set(encodings.drop(["ConTextTab", "TabSTAR-v1.1"]).map(tuple)) == {tuple(sorted(ENCODINGS))}


def test_the_space_per_dataset_results_follow_the_results(runs):
    shown = pd.read_csv(os.path.join(SPACE_DATA_DIR, "per_dataset.csv"))
    assert np.allclose(shown[ResultKey.TEST_ERROR], per_dataset(runs)[ResultKey.TEST_ERROR], rtol=1e-5, atol=1e-6)


def test_the_space_cost_follows_the_results(runs):
    shown = pd.read_csv(os.path.join(SPACE_DATA_DIR, "cost.csv")).set_index(ENTRY)
    computed = cost(runs).set_index(ENTRY).loc[shown.index]
    for column in ["train_s_per_1k", "inference_s_per_1k", "finetune_s_per_1k", "text_encode_s_per_1k",
                   "image_encode_s_per_1k"]:
        assert np.allclose(shown[column], computed[column], atol=1e-3)


def test_raw_text_models_pay_no_text_encoding(runs):
    encode = cost(runs).set_index(ENTRY)
    assert (encode.loc[["ConTextTab", "TabSTAR-v1.1"], "text_encode_s_per_1k"] == 0).all()
    assert (encode.drop(["ConTextTab", "TabSTAR-v1.1"])["text_encode_s_per_1k"] > 0).all()
    assert (encode["image_encode_s_per_1k"] > 0).all()


def test_every_model_has_a_developer_logo_and_link(runs):
    models = pd.read_csv(os.path.join(SPACE_DATA_DIR, "models.csv"))
    assert sorted(models[ResultKey.MODEL]) == sorted(runs[ResultKey.MODEL].unique())
    logos = os.path.join(os.path.dirname(SPACE_DATA_DIR), "assets", "logos")
    assert all(os.path.isfile(os.path.join(logos, logo)) for logo in models["logo"])
    assert models["link"].str.startswith("https://").all()


def test_only_tar_entries_pay_fine_tuning(runs):
    finetune = cost(runs).set_index("encoding")["finetune_s_per_1k"]
    assert (finetune.loc["tar"] > 0).all() and (finetune.drop("tar") == 0).all()
