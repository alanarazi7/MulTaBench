import zipfile

import pytest

from multabench.datasets import hub
from multabench.datasets.all_datasets import MulTaBenchDatasetID, is_benchmark_dataset, is_image_dataset
from multabench.datasets.hub import HF_REPOS


def test_every_dataset_has_one_repo():
    assert set(HF_REPOS) == set(MulTaBenchDatasetID)
    assert len(set(HF_REPOS.values())) == len(HF_REPOS)


@pytest.mark.parametrize("dataset_id", list(MulTaBenchDatasetID))
def test_repo_name_matches_the_dataset(dataset_id):
    tier, modality, task = HF_REPOS[dataset_id].split("-")[:3]
    assert tier == ("core" if is_benchmark_dataset(dataset_id) else "extended")
    assert modality == ("img" if is_image_dataset(dataset_id) else "text")
    assert task == ("reg" if dataset_id.name.startswith("REG_") else "cls")


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
