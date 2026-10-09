import numpy as np
import pytest
from pandas import DataFrame

from multabench.baselines.preprocessing import text_embeddings
from multabench.baselines.preprocessing.text_embeddings import fit_text_encoders, transform_text_features
from multabench.datasets.all_datasets import MulTaBenchDatasetID, dataset_modality
from multabench.dino.constants import DINOV3_SMALL, ImageEncoder
from multabench.e5.constants import E5_SMALL_V2, TF_IDF, TextEncoder
from multabench.embeddings.hub import (CACHED_ENCODERS, META_JSON, cached_encoder, embeddings_dir, embeddings_repo_id,
                                       embeds_dataset, parse_encoder, read_embeddings, select_rows, write_embeddings)
from multabench.embeddings.jobs import EmbeddingJob, list_jobs, pending_jobs

TEXT = MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING
IMAGE_TEXT = MulTaBenchDatasetID.MUL_IMAGE_PETFINDER


def test_one_repo_per_encoder():
    assert embeddings_repo_id(TextEncoder.E5_SMALL) == "multabench/embeddings-e5-small"
    assert embeddings_repo_id(ImageEncoder.DINO_SMALL) == "multabench/embeddings-dino-small"
    assert embeddings_dir(TEXT) == TEXT.value


def test_parse_encoder():
    assert parse_encoder("dino-small") is ImageEncoder.DINO_SMALL
    for uncached in ("tfidf", "e5-small-tar", "dino-small-tar"):
        with pytest.raises(ValueError):
            parse_encoder(uncached)


def test_cached_encoder():
    assert cached_encoder(E5_SMALL_V2, tuned=False) is TextEncoder.E5_SMALL
    assert cached_encoder(DINOV3_SMALL, tuned=False) is ImageEncoder.DINO_SMALL
    assert cached_encoder(E5_SMALL_V2, tuned=True) is None
    assert cached_encoder(TF_IDF, tuned=False) is None


@pytest.mark.parametrize("dataset_id", list(MulTaBenchDatasetID))
def test_encoders_follow_the_benchmark_runs(dataset_id):
    modality = dataset_modality(dataset_id)
    assert embeds_dataset(TextEncoder.E5_SMALL, dataset_id) == modality.has_text
    assert embeds_dataset(ImageEncoder.DINO_SMALL, dataset_id) == modality.has_images


def test_round_trip_and_row_selection(tmp_path):
    rows = np.array([7, 2, 9, 4])
    embeddings = {"title": np.arange(8, dtype=np.float32).reshape(4, 2), "body text": np.ones((4, 3))}
    write_embeddings(str(tmp_path), rows=rows, embeddings=embeddings, meta={"encode_seconds": {"title": 1.5}})
    read_rows, read, meta = read_embeddings(str(tmp_path))
    assert read_rows.tolist() == rows.tolist()
    assert set(read) == {"title", "body text"} and read["body text"].dtype == np.float32
    assert meta == {"encode_seconds": {"title": 1.5}}
    selected = select_rows(read_rows, read, idx=np.array([9, 2]))
    assert selected["title"].tolist() == [[4, 5], [2, 3]]
    with pytest.raises(KeyError):
        select_rows(read_rows, read, idx=np.array([2, 3]))


def test_jobs_cover_each_dataset_once(tmp_path):
    jobs = list_jobs(encoders=list(CACHED_ENCODERS), datasets=[TEXT, IMAGE_TEXT])
    assert jobs == [EmbeddingJob(TextEncoder.E5_SMALL, TEXT), EmbeddingJob(TextEncoder.E5_SMALL, IMAGE_TEXT),
                    EmbeddingJob(ImageEncoder.DINO_SMALL, IMAGE_TEXT)]
    assert jobs[0].command(python="python", output_dir="out") == (
        "python -m multabench.embeddings.embed --encoder e5-small --dataset_name BIN_TEXT_FAKE_JOB_POSTING --output_dir out")
    done = tmp_path / jobs[0].path
    done.mkdir(parents=True)
    (done / META_JSON).touch()
    assert pending_jobs(jobs, output_dir=str(tmp_path)) == jobs[1:]


def test_text_features_read_the_cache_by_row(monkeypatch):
    rng = np.random.default_rng(0)
    cache = rng.normal(size=(60, 32)).astype(np.float32)
    monkeypatch.setattr(text_embeddings, "cached_embeddings", lambda encoder, dataset: (np.arange(60), {"title": cache}))
    monkeypatch.setattr(text_embeddings, "get_vanilla_e5", lambda *args, **kwargs: pytest.fail("loaded E5 for a cached column"))
    x = DataFrame({"title": [f"t{i}" for i in range(60)]})
    train_idx, test_idx = rng.permutation(60)[:45], np.arange(60)[::4]
    encoders, train_embeddings = fit_text_encoders(x.iloc[train_idx], text_features={"title"}, device="cpu", dataset=TEXT)
    np.testing.assert_array_equal(train_embeddings["title"], cache[train_idx])
    x_test = transform_text_features(x.iloc[test_idx], text_encoders=encoders, device="cpu")
    np.testing.assert_allclose(x_test.to_numpy(), encoders["title"].encoder.transform(cache[test_idx]), rtol=1e-6)
