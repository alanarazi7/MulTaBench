import pytest

from multabench import Modality, SupervisedTask, Tier, list_datasets
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.hub import hf_repo_id
from multabench.datasets.info import dataset_info
from multabench.datasets.revisions import revision
from multabench.embeddings.hub import CACHED_ENCODERS, embeddings_repo_id
from multabench.leaderboard.export import dataset_tier


def test_every_dataset_is_listed_once():
    assert [i.id for i in list_datasets()] == list(MulTaBenchDatasetID)


def test_the_filters_partition_the_datasets():
    for values, key in [(Modality, "modality"), (SupervisedTask, "task"), (Tier, "tier")]:
        assert sum(len(list_datasets(**{key: v})) for v in values) == len(MulTaBenchDatasetID)


def test_filters_take_strings():
    infos = list_datasets(modality="text", task="binary", tier="core")
    assert infos == list_datasets(modality=Modality.TEXT, task=SupervisedTask.BINARY, tier=Tier.CORE)
    assert MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING in [i.id for i in infos]


@pytest.mark.parametrize("dataset_id", list(MulTaBenchDatasetID))
def test_info_matches_the_repo(dataset_id):
    info = dataset_info(dataset_id.name)
    assert info.hf_repo == hf_repo_id(dataset_id) and info.tier == dataset_tier(dataset_id.name)


def test_every_repo_is_pinned_to_a_commit():
    repos = [hf_repo_id(d) for d in MulTaBenchDatasetID] + [embeddings_repo_id(e) for e in CACHED_ENCODERS]
    for repo in repos:
        assert len(revision(repo)) == 40 and int(revision(repo), 16) >= 0
