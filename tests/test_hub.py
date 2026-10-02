import zipfile

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


def test_image_shards_restore_the_images_folder(tmp_path):
    names = ["images/a.jpg", "images/b.png", "images/sub/c.jpg"]
    shards = {"images-00000.zip": names[:2], "images-00001.zip": names[2:]}
    for shard, shard_names in shards.items():
        with zipfile.ZipFile(tmp_path / shard, "w") as z:
            for name in shard_names:
                z.writestr(name, bytes([names.index(name)]) * 10)
    hub.extract_images(str(tmp_path))
    for i, name in enumerate(names):
        assert (tmp_path / name).read_bytes() == bytes([i]) * 10
    assert sorted(p.name for p in tmp_path.iterdir()) == ["images", "images-00000.zip", "images-00001.zip"]
