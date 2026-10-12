import numpy as np
import pytest
from pandas import DataFrame

from multabench.baselines.preprocessing import text_embeddings
from multabench.baselines.preprocessing.nulls import MISSING_VALUE
from multabench.baselines.preprocessing.text_embeddings import fit_text_encoders, transform_text_features
from multabench.datasets.all_datasets import MulTaBenchDatasetID, dataset_modality
from multabench.dino.constants import DINOV3_SMALL, IMAGE_ENCODERS, ImageEncoder
from multabench.e5.constants import E5_SMALL_V2, TEXT_ENCODERS, TF_IDF, TextEncoder, format_e5_passage
from multabench.embeddings import hub
from multabench.embeddings.hub import (CACHED_ENCODERS, META_JSON, CachedEmbeddings, cached_encoder, download_embeddings,
                                       embeddings_dir, embeddings_repo_id, embeds_dataset, encoder_inputs, input_values, parse_encoder,
                                       read_embeddings, reads_cache, write_embeddings)
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


@pytest.mark.parametrize("dataset_id", list(MulTaBenchDatasetID))
def test_encoders_follow_the_benchmark_runs(dataset_id):
    modality = dataset_modality(dataset_id)
    assert embeds_dataset(TextEncoder.E5_SMALL, dataset_id) == modality.has_text
    assert embeds_dataset(ImageEncoder.DINO_SMALL, dataset_id) == modality.has_images


def test_round_trip_and_lookup(tmp_path):
    inputs = ["passage: title: red", "passage: title: blue", "passage: body: a longer text, with ünïcode"]
    embeddings = np.arange(6, dtype=np.float32).reshape(3, 2)
    write_embeddings(str(tmp_path), inputs=inputs, embeddings=embeddings, meta={"features": ["title", "body"]})
    cache = read_embeddings(str(tmp_path))
    assert cache.inputs == inputs and cache.embeddings.dtype == np.float32
    assert cache.meta == {"features": ["title", "body"]}
    assert cache.lookup(["passage: title: blue", "passage: title: red", "passage: title: blue"], "title").tolist() == [
        [2, 3], [0, 1], [2, 3]]


def test_an_input_missing_from_the_cache_raises(tmp_path):
    write_embeddings(str(tmp_path), inputs=["passage: title: red"], embeddings=np.zeros((1, 2)), meta={})
    with pytest.raises(KeyError, match="1 of 2 inputs of 'title' have no cached embedding"):
        read_embeddings(str(tmp_path)).lookup(["passage: title: red", "passage: title: Red"], "title")


def test_inputs_are_what_the_encoder_gets():
    x = DataFrame({"title": ["a", None, np.nan], "image": ["img/1.jpg", "img/2.jpg", "img/1.jpg"]})
    values = input_values(x, TextEncoder.E5_SMALL, "title")
    assert values == ["a", MISSING_VALUE, MISSING_VALUE]
    assert encoder_inputs(TextEncoder.E5_SMALL, "title", values)[0] == format_e5_passage("title", col_val="a")
    assert encoder_inputs(ImageEncoder.DINO_SMALL, "image", input_values(x, ImageEncoder.DINO_SMALL, "image")) == [
        "img/1.jpg", "img/2.jpg", "img/1.jpg"]


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


def test_cached_encoder():
    assert cached_encoder(E5_SMALL_V2, tuned=False) is TextEncoder.E5_SMALL
    assert cached_encoder(DINOV3_SMALL, tuned=False) is ImageEncoder.DINO_SMALL
    assert cached_encoder(E5_SMALL_V2, tuned=True) is None
    assert cached_encoder(TF_IDF, tuned=False) is None


def test_runs_read_only_cached_text_embeddings():
    assert reads_cache(TEXT_ENCODERS[TextEncoder.E5_SMALL])
    assert not reads_cache(TEXT_ENCODERS[TextEncoder.E5_SMALL_TAR])
    assert not reads_cache(TEXT_ENCODERS[TextEncoder.TFIDF])
    assert not reads_cache(IMAGE_ENCODERS[ImageEncoder.DINO_SMALL])


def test_missing_cached_embeddings_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    (tmp_path / embeddings_dir(TEXT)).mkdir()
    (tmp_path / embeddings_dir(TEXT) / META_JSON).touch()
    with pytest.raises(FileNotFoundError, match="embeddings.npz"):
        download_embeddings(TextEncoder.E5_SMALL, TEXT)


def test_text_features_look_their_texts_up_in_the_cache(monkeypatch):
    rng = np.random.default_rng(0)
    texts = [f"t{i % 40}" for i in range(60)]
    distinct = sorted(set(texts))
    vectors = rng.normal(size=(len(distinct), 32)).astype(np.float32)
    cache = CachedEmbeddings(inputs=encoder_inputs(TextEncoder.E5_SMALL, "title", distinct), embeddings=vectors,
                             meta={"features": ["title"]})
    monkeypatch.setattr(text_embeddings, "cached_embeddings", lambda encoder, dataset: cache)
    monkeypatch.setattr(text_embeddings, "get_vanilla_e5", lambda *args, **kwargs: pytest.fail("loaded E5 for a cached column"))
    x = DataFrame({"title": texts})
    expected = vectors[[distinct.index(t) for t in texts]]
    train_idx, test_idx = rng.permutation(60)[:45], np.arange(60)[::4]
    encoders, train_embeddings = fit_text_encoders(x.iloc[train_idx], text_features={"title"}, device="cpu", dataset=TEXT)
    np.testing.assert_array_equal(train_embeddings["title"], expected[train_idx])
    x_test = transform_text_features(x.iloc[test_idx], text_encoders=encoders, device="cpu")
    np.testing.assert_allclose(x_test.to_numpy(), encoders["title"].encoder.transform(expected[test_idx]), rtol=1e-6)
    with pytest.raises(KeyError, match="no cached embedding"):
        transform_text_features(DataFrame({"title": ["a text the cache never saw"]}), text_encoders=encoders, device="cpu")
