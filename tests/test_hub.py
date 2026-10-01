import pytest

from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.hub import hf_repo_id


@pytest.mark.parametrize("dataset_id, repo_id", [
    (MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES, "multabench/core-text-reg-montgomery-salaries"),
    (MulTaBenchDatasetID.BIN_TEXT_KICKSTARTER_FUNDING, "multabench/core-text-cls-kickstarter-funding"),
    (MulTaBenchDatasetID.MUL_IMAGE_PETFINDER, "multabench/core-img-cls-petfinder"),
    (MulTaBenchDatasetID.REG_TEXT_AIRBNB_SEATTLE, "multabench/extended-text-reg-airbnb-seattle"),
])
def test_hf_repo_id(dataset_id, repo_id):
    assert hf_repo_id(dataset_id) == repo_id


def test_hf_repo_ids_are_unique():
    repo_ids = [hf_repo_id(d) for d in MulTaBenchDatasetID]
    assert len(repo_ids) == len(set(repo_ids))
