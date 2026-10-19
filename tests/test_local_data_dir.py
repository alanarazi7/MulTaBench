import json
import os
import stat
from os.path import join

import numpy as np
import pandas as pd
import pytest

from multabench.benchmark import load
from multabench.benchmark.load import load_embeddings, load_split, load_target
from multabench.benchmark.splits import Size
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.e5.constants import TextEncoder
from multabench.embeddings import hub
from multabench.embeddings.hub import embeddings_dir, encoder_inputs, input_values, repo_name, write_embeddings

TEXT = MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING
IMAGE = MulTaBenchDatasetID.BIN_IMAGE_OASIS_ALZHEIMERS


def _write_dataset(data_dir, dataset_id=TEXT):
    titles = [f"t{i % 7}" if i % 9 else None for i in range(60)]
    x = pd.DataFrame({"title": titles, "label": [i % 2 for i in range(60)]})
    path = join(data_dir, dataset_id.value)
    os.makedirs(path)
    x.to_parquet(join(path, "data.parquet"))
    with open(join(path, "metadata.json"), "w") as f:
        json.dump({"target": "label", "image_col": None}, f)
    return x


def _write_embeddings(data_dir, x, encoder, size=None, fold=None):
    inputs = sorted(set(encoder_inputs(encoder, "title", input_values(x, encoder, "title"))))
    vectors = np.random.default_rng(0).normal(size=(len(inputs), 4)).astype(np.float32)
    path = join(data_dir, repo_name(encoder), embeddings_dir(TEXT, size=size, fold=fold))
    write_embeddings(path, inputs=inputs, embeddings=vectors, meta={"features": ["title"]})


@pytest.fixture
def data_dir(tmp_path):
    x = _write_dataset(str(tmp_path))
    _write_embeddings(str(tmp_path), x, TextEncoder.E5_SMALL)
    _write_embeddings(str(tmp_path), x, TextEncoder.E5_SMALL_TAR, size=Size.TEN_K, fold=2)
    return str(tmp_path)


def _hub_serves(monkeypatch, data_dir):
    """snapshot_download returns the local repo folder, as if the Hub held the same files."""
    serve = lambda repo_id, **kwargs: join(data_dir, repo_id.split("/")[-1])
    monkeypatch.setattr(load, "snapshot_download", serve)
    monkeypatch.setattr(hub, "snapshot_download", serve)


def _no_hub(monkeypatch):
    def offline(**kwargs):
        raise AssertionError("downloaded from the Hub")
    monkeypatch.setattr(load, "snapshot_download", offline)
    monkeypatch.setattr(hub, "snapshot_download", offline)


def test_a_local_copy_gives_the_hubs_split(monkeypatch, data_dir):
    _hub_serves(monkeypatch, data_dir)
    expected = load_split(TEXT, fold=1)
    _no_hub(monkeypatch)
    split = load_split(TEXT.name, fold=1, data_dir=data_dir)
    pd.testing.assert_frame_equal(split.x_train, expected.x_train)
    pd.testing.assert_series_equal(split.y_test, expected.y_test)
    assert split.task_type == expected.task_type
    y, task_type = load_target(TEXT, data_dir=data_dir)
    pd.testing.assert_series_equal(y, pd.concat([expected.y_train, expected.y_test]).sort_index())
    assert task_type == expected.task_type


@pytest.mark.parametrize("encoder, fold", [("e5-small", 1), ("e5-small-tar", 2)])
def test_a_local_copy_gives_the_hubs_embeddings(monkeypatch, data_dir, encoder, fold):
    _hub_serves(monkeypatch, data_dir)
    expected = load_embeddings(TEXT, fold=fold, encoder=encoder)
    _no_hub(monkeypatch)
    embeddings = load_embeddings(TEXT, fold=fold, encoder=encoder, data_dir=data_dir)
    np.testing.assert_array_equal(embeddings.train["title"], expected.train["title"])
    np.testing.assert_array_equal(embeddings.test["title"], expected.test["title"])


def test_a_missing_file_names_its_path(monkeypatch, data_dir):
    _no_hub(monkeypatch)
    with pytest.raises(FileNotFoundError, match=join("embeddings-e5-small-tar", TEXT.value, "10k", "fold_0")):
        load_embeddings(TEXT, fold=0, encoder="e5-small-tar", data_dir=data_dir)
    os.remove(join(data_dir, TEXT.value, "metadata.json"))
    with pytest.raises(FileNotFoundError, match=join(TEXT.value, "metadata.json")):
        load_split(TEXT, fold=0, data_dir=data_dir)


def test_an_image_dataset_needs_its_images_extracted(monkeypatch, tmp_path):
    _no_hub(monkeypatch)
    _write_dataset(str(tmp_path), dataset_id=IMAGE)
    with pytest.raises(FileNotFoundError, match=join(IMAGE.value, "images")):
        load_split(IMAGE, fold=0, data_dir=str(tmp_path))


def test_a_read_only_copy_loads(monkeypatch, data_dir):
    _no_hub(monkeypatch)
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            os.chmod(join(root, name), stat.S_IRUSR)
        os.chmod(root, stat.S_IRUSR | stat.S_IXUSR)
    try:
        load_split(TEXT, fold=0, data_dir=data_dir)
        load_embeddings(TEXT, fold=0, encoder="e5-small", data_dir=data_dir)
    finally:
        for root, dirs, files in os.walk(data_dir):
            os.chmod(root, stat.S_IRWXU)
