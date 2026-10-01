import pytest

from multabench.datasets import hub
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


def test_image_shards_restore_the_images_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "IMAGES_PER_SHARD", 2)
    src, repo = tmp_path / "src", tmp_path / "repo"
    (src / "images" / "sub").mkdir(parents=True)
    repo.mkdir()
    names = ["images/a.jpg", "images/b.png", "images/sub/c.jpg"]
    for i, name in enumerate(names):
        (src / name).write_bytes(bytes([i]) * 10)
    assert hub.pack_images(str(src), str(repo)) == 3
    assert sorted(p.name for p in repo.iterdir()) == ["images-00000.zip", "images-00001.zip"]
    hub.extract_images(str(repo))
    for i, name in enumerate(names):
        assert (repo / name).read_bytes() == bytes([i]) * 10
    assert sorted(p.name for p in repo.iterdir()) == ["images", "images-00000.zip", "images-00001.zip"]
