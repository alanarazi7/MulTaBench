import numpy as np
import pytest

from multabench.benchmark.splits import FOLDS, MAX_TEST_ROWS, MAX_TRAIN_ROWS, REPEATS, SPLITS, get_split


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
        tests = np.concatenate([get_split(y, is_cls=True, split=repeat * FOLDS + k)[1] for k in range(FOLDS)])
        assert sorted(tests) == list(range(len(y)))


def test_train_test_ratio_is_two_to_one():
    train, test = get_split(np.random.default_rng(0).integers(0, 2, 900), is_cls=True, split=0)
    assert (len(train), len(test)) == (600, 300)


def test_stratification_survives_capping():
    y = np.array([0] * 45_000 + [1] * 5_000)
    for split in range(SPLITS):
        train, test = get_split(y, is_cls=True, split=split)
        assert abs(y[train].mean() - 0.1) < 0.005
        assert abs(y[test].mean() - 0.1) < 0.005


def test_rare_class_falls_back_to_unstratified():
    y = np.array([0] * 100 + [1] * 2)
    assert all(len(get_split(y, is_cls=True, split=s)[1]) == 34 for s in range(SPLITS))


def test_splits_are_unchanged():
    # Pinned so that a scikit-learn upgrade that changes its shuffling fails here instead of
    # silently changing every benchmark split.
    y = np.arange(12) % 2
    assert get_split(y, is_cls=True, split=0)[1].tolist() == [0, 2, 3, 5]
    assert get_split(y, is_cls=True, split=4)[1].tolist() == [5, 6, 10, 11]
    assert get_split(np.linspace(0, 1, 9), is_cls=False, split=5)[1].tolist() == [2, 6, 8]
