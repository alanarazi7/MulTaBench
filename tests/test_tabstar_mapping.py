import pytest

from multabench.baselines.tabstar_v1 import NEW2PAPER
from multabench.datasets.all_datasets import MulTaBenchDatasetID


def test_every_dataset_is_mapped():
    assert set(NEW2PAPER) == {d.name for d in MulTaBenchDatasetID}


def test_mapped_names_are_tabstar_pretraining_datasets():
    tabstar_datasets = pytest.importorskip("tabstar.tabstar_datasets")
    known = tabstar_datasets.TEXT2FOLD.keys() | tabstar_datasets.PRETRAIN2FOLD.keys()
    assert {name for name in NEW2PAPER.values() if name is not None} <= known
