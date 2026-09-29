"""Regenerate the committed splits in data/splits/. Downloads each dataset from Kaggle.

    python -m multabench.scripts.make_splits                      # all 80 datasets
    python -m multabench.scripts.make_splits BIN_TEXT_JIGSAW_TOXICITY
"""
import sys

from multabench.benchmark.load import load_multabench_dataset
from multabench.benchmark.splits import make_splits, save_splits
from multabench.datasets.all_datasets import MulTaBenchDatasetID

if __name__ == "__main__":
    names = sys.argv[1:] or [d.name for d in MulTaBenchDatasetID]
    for name in names:
        dataset = load_multabench_dataset(MulTaBenchDatasetID[name])
        folds = make_splits(dataset.y.to_numpy(), is_cls=dataset.is_cls)
        path = save_splits(name, folds, n_rows=len(dataset.y))
        sizes = ", ".join(f"{len(f.train_idx)}/{len(f.test_idx)}" for f in folds)
        print(f"{name}: {len(dataset.y)} rows, train/test per fold {sizes} -> {path}")
