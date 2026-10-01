from tabstar.tabstar_datasets import TEXT2FOLD

from multabench.baselines.tabstar_v1 import NEW2PAPER


def test_paper_datasets_are_known_to_tabstar():
    unknown = [d for d in NEW2PAPER.values() if d is not None and d not in TEXT2FOLD]
    assert not unknown
