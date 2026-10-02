import zipfile

import pytest

from multabench.datasets import hub
from multabench.datasets.all_datasets import MulTaBenchDatasetID, is_benchmark_dataset, is_image_dataset
from multabench.datasets.hub import hf_repo_id


def test_40_benchmark_datasets():
    assert sum(is_benchmark_dataset(d) for d in MulTaBenchDatasetID) == 40


@pytest.mark.parametrize("dataset_id", list(MulTaBenchDatasetID))
def test_repo_name_matches_the_dataset(dataset_id):
    tier, modality, task = dataset_id.value.split("-")[:3]
    assert tier in ("core", "extended")
    assert modality == ("img" if is_image_dataset(dataset_id) else "text")
    assert task == ("reg" if dataset_id.name.startswith("REG_") else "cls")
    assert hf_repo_id(dataset_id) == f"multabench/{dataset_id.value}"


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
