"""Outer train/test splits: TabArena's repeated 3-fold scheme (67/33), with 2 repeats.

Split k is fold k % 3 of repeat k // 3, stratified for classification. Each split's train and test
parts are then capped at 10K and 5K rows, keeping the class balance. Indices are row positions.
"""
from typing import Tuple

import numpy as np
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, train_test_split

FOLDS = 3
REPEATS = 2
SPLITS = FOLDS * REPEATS
SEED = 42
MAX_TRAIN_ROWS = 10_000
MAX_TEST_ROWS = 5_000


def get_split(y: np.ndarray, is_cls: bool, split: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    stratify = is_cls and np.unique(y, return_counts=True)[1].min() >= FOLDS
    cv_cls = RepeatedStratifiedKFold if stratify else RepeatedKFold
    cv = cv_cls(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
    train, test = list(cv.split(np.zeros(len(y)), y))[split]
    train = _cap(train, y=y, is_cls=is_cls, size=MAX_TRAIN_ROWS, seed=SEED + split)
    test = _cap(test, y=y, is_cls=is_cls, size=MAX_TEST_ROWS, seed=SEED + split)
    return np.sort(train), np.sort(test)


def _cap(idx: np.ndarray, y: np.ndarray, is_cls: bool, size: int, seed: int) -> np.ndarray:
    if len(idx) <= size:
        return idx
    stratify = y[idx] if is_cls and np.unique(y[idx], return_counts=True)[1].min() >= 2 else None
    kept, _ = train_test_split(idx, train_size=size, random_state=seed, stratify=stratify)
    return kept
