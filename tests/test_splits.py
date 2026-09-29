import numpy as np
import pytest

from multabench.benchmark.splits import FOLDS, MAX_TEST_ROWS, MAX_TRAIN_ROWS, Fold, load_splits, make_splits, save_splits
from multabench.datasets.all_datasets import MulTaBenchDatasetID


@pytest.mark.parametrize("n, is_cls", [(300, True), (60_000, True), (30_000, False)])
def test_make_splits_caps_and_disjointness(n, is_cls):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, n) if is_cls else rng.normal(size=n)
    folds = make_splits(y, is_cls=is_cls)
    assert len(folds) == FOLDS
    tests = np.concatenate([f.test_idx for f in folds])
    assert len(set(tests)) == len(tests)
    for f in folds:
        assert not set(f.train_idx) & set(f.test_idx)
        assert len(f.train_idx) <= MAX_TRAIN_ROWS and len(f.test_idx) <= MAX_TEST_ROWS
    if n <= MAX_TRAIN_ROWS:
        assert len(tests) == n


def test_make_splits_is_deterministic():
    y = np.random.default_rng(1).integers(0, 2, 5_000)
    a, b = make_splits(y, is_cls=True), make_splits(y, is_cls=True)
    assert all(np.array_equal(x.train_idx, z.train_idx) and np.array_equal(x.test_idx, z.test_idx) for x, z in zip(a, b))


def test_stratification_survives_capping():
    y = np.array([0] * 45_000 + [1] * 5_000)
    for f in make_splits(y, is_cls=True):
        assert abs(y[f.train_idx].mean() - 0.1) < 0.005
        assert abs(y[f.test_idx].mean() - 0.1) < 0.01


def test_rare_class_falls_back_to_unstratified():
    y = np.array([0] * 100 + [1] * 2)
    assert len(make_splits(y, is_cls=True)) == FOLDS


def test_save_load_roundtrip(tmp_path):
    folds = [Fold(np.arange(8), np.arange(8, 10)), Fold(np.arange(2, 10), np.arange(2))]
    save_splits("X", folds, n_rows=10, splits_dir=str(tmp_path))
    back = load_splits("X", n_rows=10, splits_dir=str(tmp_path))
    assert [(f.train_idx.tolist(), f.test_idx.tolist()) for f in back] == [
        (f.train_idx.tolist(), f.test_idx.tolist()) for f in folds
    ]
    with pytest.raises(AssertionError):
        load_splits("X", n_rows=11, splits_dir=str(tmp_path))


@pytest.mark.parametrize("dataset", [d.name for d in MulTaBenchDatasetID])
def test_committed_splits(dataset):
    folds = load_splits(dataset)
    assert len(folds) == FOLDS
    for f in folds:
        assert not set(f.train_idx) & set(f.test_idx)
        assert len(f.train_idx) <= MAX_TRAIN_ROWS and len(f.test_idx) <= MAX_TEST_ROWS
