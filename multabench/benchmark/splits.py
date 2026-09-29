"""Fixed outer splits, committed in data/splits/<dataset>.npz.

Each dataset gets 5 folds over all its rows (stratified for classification), and each fold's train
and test parts are then capped at 10K and 2.5K rows. Indices are positions in the dataset's data.csv.
"""
from dataclasses import dataclass
from os import makedirs
from os.path import dirname, join

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

FOLDS = 5
SEED = 42
MAX_TRAIN_ROWS = 10_000
MAX_TEST_ROWS = 2_500
SPLITS_DIR = join(dirname(dirname(dirname(__file__))), "data", "splits")


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def splits_path(dataset_name: str, splits_dir: str = SPLITS_DIR) -> str:
    return join(splits_dir, f"{dataset_name}.npz")


def load_splits(dataset_name: str, n_rows: int | None = None, splits_dir: str = SPLITS_DIR) -> list[Fold]:
    with np.load(splits_path(dataset_name, splits_dir)) as z:
        if n_rows is not None:
            assert int(z["n_rows"]) == n_rows, f"{dataset_name}: splits are for {int(z['n_rows'])} rows, data has {n_rows}"
        return [Fold(train_idx=z[f"train_{k}"], test_idx=z[f"test_{k}"]) for k in range(int(z["n_folds"]))]


def save_splits(dataset_name: str, folds: list[Fold], n_rows: int, splits_dir: str = SPLITS_DIR) -> str:
    makedirs(splits_dir, exist_ok=True)
    arrays = {"n_folds": np.int64(len(folds)), "n_rows": np.int64(n_rows)}
    for k, f in enumerate(folds):
        arrays[f"train_{k}"] = f.train_idx.astype(np.int32)
        arrays[f"test_{k}"] = f.test_idx.astype(np.int32)
    path = splits_path(dataset_name, splits_dir)
    np.savez_compressed(path, **arrays)
    return path


def make_splits(y: np.ndarray, is_cls: bool) -> list[Fold]:
    y = np.asarray(y)
    stratify = is_cls and _can_stratify(y)
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED) if stratify else KFold(FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for k, (train, test) in enumerate(cv.split(np.arange(len(y)), y if stratify else None)):
        train = _cap(train, y=y, is_cls=is_cls, size=MAX_TRAIN_ROWS, seed=SEED + k)
        test = _cap(test, y=y, is_cls=is_cls, size=MAX_TEST_ROWS, seed=SEED + k)
        folds.append(Fold(train_idx=np.sort(train), test_idx=np.sort(test)))
    return folds


def _can_stratify(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return counts.min() >= FOLDS


def _cap(idx: np.ndarray, y: np.ndarray, is_cls: bool, size: int, seed: int) -> np.ndarray:
    if len(idx) <= size:
        return idx
    stratify = y[idx] if is_cls and np.unique(y[idx], return_counts=True)[1].min() >= 2 else None
    kept, _ = train_test_split(idx, train_size=size, random_state=seed, stratify=stratify)
    return kept
