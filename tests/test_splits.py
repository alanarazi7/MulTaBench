import numpy as np
import pandas as pd
import pytest

from multabench.benchmark.load import split_dataset
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.objects import MultimodalDataset, SupervisedTask
from multabench.benchmark.splits import FOLDS, MAX_TEST_ROWS, MAX_TRAIN_ROWS, REPEATS, SIZE_FULL, SPLITS, get_split


@pytest.mark.parametrize("n, is_cls", [(300, True), (60_000, True), (30_000, False)])
def test_splits_are_disjoint_and_capped(n, is_cls):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, n) if is_cls else rng.normal(size=n)
    for split in range(SPLITS):
        train, test = get_split(y, is_cls=is_cls, split=split)
        assert not set(train) & set(test)
        assert len(train) <= MAX_TRAIN_ROWS and len(test) <= MAX_TEST_ROWS


def test_each_repeat_tests_every_row_once():
    y = np.random.default_rng(0).integers(0, 3, 900)
    for repeat in range(REPEATS):
        tested_rows = []
        for fold in range(FOLDS):
            _, test = get_split(y, is_cls=True, split=repeat * FOLDS + fold)
            tested_rows.extend(test)
        assert sorted(tested_rows) == list(range(len(y)))


def test_train_test_ratio_is_two_to_one():
    train, test = get_split(np.arange(900) % 2, is_cls=True, split=0)
    assert (len(train), len(test)) == (600, 300)


def test_full_size_keeps_every_row():
    train, test = get_split(np.random.default_rng(0).normal(size=30_000), is_cls=False, split=0, size=SIZE_FULL)
    assert (len(train), len(test)) == (20_000, 10_000)


def test_stratification_survives_capping():
    y = np.array([0] * 45_000 + [1] * 5_000)
    for split in range(SPLITS):
        train, test = get_split(y, is_cls=True, split=split)
        assert abs(y[train].mean() - 0.1) < 0.005
        assert abs(y[test].mean() - 0.1) < 0.005


def test_rare_class_falls_back_to_unstratified():
    y = np.array([0] * 100 + [1] * 2)
    for split in range(SPLITS):
        _, test = get_split(y, is_cls=True, split=split)
        assert len(test) == 34


def test_splits_are_unchanged():
    # Pinned so that a scikit-learn upgrade that changes its shuffling fails here instead of
    # silently changing every benchmark split.
    y = np.arange(12) % 2
    assert get_split(y, is_cls=True, split=0)[1].tolist() == [0, 2, 3, 5]
    assert get_split(y, is_cls=True, split=4)[1].tolist() == [5, 6, 10, 11]
    assert get_split(np.linspace(0, 1, 9), is_cls=False, split=5)[1].tolist() == [2, 6, 8]


def test_split_dataset_matches_get_split():
    y = pd.Series(np.arange(300) % 3)
    x = pd.DataFrame({"feature": np.arange(300)})
    dataset = MultimodalDataset(x=x, y=y, task_type=SupervisedTask.MULTICLASS,
                                dataset_id=MulTaBenchDatasetID.MUL_IMAGE_PETFINDER)
    split = split_dataset(dataset, fold=1)
    train, test = get_split(y.to_numpy(), is_cls=True, split=1)
    assert split.x_train["feature"].tolist() == train.tolist()
    assert split.y_test.tolist() == y.iloc[test].tolist()
