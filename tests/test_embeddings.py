import numpy as np
import pytest
from pandas import DataFrame

from multabench.baselines.preprocessing.nulls import MISSING_VALUE
from multabench.benchmark.splits import Size
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.embeddings import hub
from multabench.embeddings.hub import (CACHED_ENCODERS, META_JSON, CachedEmbeddings, cached_encoder, download_embeddings,
                                       embeddings_dir, embeddings_repo_id, embeds_dataset, encoder_inputs, input_values, parse_encoder,
                                       read_embeddings, reads_cache, write_embeddings)
from multabench.embeddings.jobs import EmbeddingJob, list_jobs, pending_jobs
from multabench.encoders import column_encoders
from multabench.encoders.column_encoders import transform_columns
from multabench.encoders.constants import IMAGE_ENCODERS, ImageEncoder, TEXT_ENCODERS, TF_IDF, TextEncoder
from multabench.encoders.dino import dino_encoder
from multabench.encoders.dino.constants import DINOV3_LARGE, DINOV3_SMALL
from multabench.encoders.e5 import e5_encoder
from multabench.encoders.e5.constants import E5_SMALL_V2, format_e5_passage
from multabench.encoders.e5.e5_encoder import E5ColumnEncoder
from multabench.encoders.registry import ENCODER_CLASSES, make_encoder

TEXT = MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING
IMAGE_TEXT = MulTaBenchDatasetID.MUL_IMAGE_PETFINDER


def test_one_repo_per_encoder():
    assert embeddings_repo_id(TextEncoder.E5_SMALL) == "multabench/embeddings-e5-small"
    assert embeddings_repo_id(ImageEncoder.DINO_SMALL) == "multabench/embeddings-dino-small"
    assert embeddings_dir(TEXT) == TEXT.value


def test_every_encoder_has_a_class():
    assert set(ENCODER_CLASSES) == set(TextEncoder) | set(ImageEncoder)


def test_tuned_encoders_are_cached_per_fold():
    assert embeddings_repo_id(TextEncoder.E5_SMALL_TAR) == "multabench/embeddings-e5-small-tar"
    assert embeddings_dir(TEXT, size=Size.TEN_K, fold=3) == f"{TEXT.value}/10k/fold_3"


def test_parse_encoder():
    assert parse_encoder("dino-small") is ImageEncoder.DINO_SMALL
    assert parse_encoder("e5-small-tar") is TextEncoder.E5_SMALL_TAR
    assert parse_encoder("e5-large") is TextEncoder.E5_LARGE
    with pytest.raises(ValueError):
        parse_encoder("tfidf")


@pytest.mark.parametrize("dataset_id", list(MulTaBenchDatasetID))
def test_encoders_follow_the_benchmark_runs(dataset_id):
    modality = dataset_modality(dataset_id)
    assert embeds_dataset(TextEncoder.E5_SMALL, dataset_id) == modality.has_text
    assert embeds_dataset(ImageEncoder.DINO_SMALL, dataset_id) == modality.has_images
    assert embeds_dataset(TextEncoder.E5_SMALL_TAR, dataset_id) == (modality == Modality.TEXT)
    assert embeds_dataset(ImageEncoder.DINO_SMALL_TAR, dataset_id) == modality.has_images
    assert embeds_dataset(TextEncoder.E5_LARGE, dataset_id) == modality.has_text
    assert embeds_dataset(ImageEncoder.DINO_LARGE, dataset_id) == modality.has_images
    assert embeds_dataset(TextEncoder.E5_LARGE_TAR, dataset_id) == (modality == Modality.TEXT)
    assert embeds_dataset(ImageEncoder.DINO_LARGE_TAR, dataset_id) == modality.has_images


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


def test_frozen_jobs_cover_each_dataset_once(tmp_path):
    jobs = list_jobs(encoders=[TextEncoder.E5_SMALL, ImageEncoder.DINO_SMALL], datasets=[TEXT, IMAGE_TEXT])
    assert jobs == [EmbeddingJob(TextEncoder.E5_SMALL, TEXT), EmbeddingJob(TextEncoder.E5_SMALL, IMAGE_TEXT),
                    EmbeddingJob(ImageEncoder.DINO_SMALL, IMAGE_TEXT)]
    assert jobs[0].command(python="python", output_dir="out") == (
        "python -m multabench.embeddings.embed --encoder e5-small --dataset_name BIN_TEXT_FAKE_JOB_POSTING --output_dir out")
    done = tmp_path / jobs[0].path
    done.mkdir(parents=True)
    (done / META_JSON).touch()
    assert pending_jobs(jobs, output_dir=str(tmp_path)) == jobs[1:]


def test_tuned_jobs_cover_each_fold(tmp_path):
    jobs = list_jobs(encoders=[ImageEncoder.DINO_SMALL_TAR], datasets=[TEXT, IMAGE_TEXT], folds=[0, 4])
    assert jobs == [EmbeddingJob(ImageEncoder.DINO_SMALL_TAR, IMAGE_TEXT, Size.TEN_K, 0),
                    EmbeddingJob(ImageEncoder.DINO_SMALL_TAR, IMAGE_TEXT, Size.TEN_K, 4)]
    assert jobs[1].path == f"embeddings-dino-small-tar/{IMAGE_TEXT.value}/10k/fold_4"
    assert jobs[1].command(python="python", output_dir="out") == (
        "python -m multabench.embeddings.embed --encoder dino-small-tar --dataset_name MUL_IMAGE_PETFINDER --size 10k "
        "--fold 4 --output_dir out")
    assert len(list_jobs(encoders=list(CACHED_ENCODERS), datasets=[IMAGE_TEXT])) == 1 + 1 + 6 + 1 + 1 + 6 + 1


def test_cached_encoder():
    assert cached_encoder(E5_SMALL_V2, tuned=False) is TextEncoder.E5_SMALL
    assert cached_encoder(DINOV3_SMALL, tuned=False) is ImageEncoder.DINO_SMALL
    assert cached_encoder(E5_SMALL_V2, tuned=True) is TextEncoder.E5_SMALL_TAR
    assert cached_encoder(DINOV3_LARGE, tuned=True) is ImageEncoder.DINO_LARGE_TAR
    assert cached_encoder(TF_IDF, tuned=False) is None


def test_runs_read_the_frozen_and_tuned_encoders_from_the_cache():
    for encoder in (TextEncoder.E5_SMALL, TextEncoder.E5_SMALL_TAR, TextEncoder.E5_LARGE_TAR):
        assert reads_cache(TEXT_ENCODERS[encoder])
    for encoder in (ImageEncoder.DINO_SMALL, ImageEncoder.DINO_SMALL_TAR, ImageEncoder.DINO_LARGE_TAR):
        assert reads_cache(IMAGE_ENCODERS[encoder])
    assert not reads_cache(TEXT_ENCODERS[TextEncoder.TFIDF])
    assert reads_cache(TEXT_ENCODERS[TextEncoder.E5_LARGE]) and reads_cache(IMAGE_ENCODERS[ImageEncoder.DINO_LARGE])


def test_missing_cached_embeddings_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    (tmp_path / embeddings_dir(TEXT)).mkdir()
    (tmp_path / embeddings_dir(TEXT) / META_JSON).touch()
    with pytest.raises(FileNotFoundError, match="embeddings.npz"):
        download_embeddings(TextEncoder.E5_SMALL, TEXT)
    with pytest.raises(ValueError, match="per fold"):
        download_embeddings(TextEncoder.E5_SMALL_TAR, TEXT)


def test_text_features_look_their_texts_up_in_the_cache(monkeypatch):
    rng = np.random.default_rng(0)
    texts = [f"t{i % 40}" for i in range(60)]
    distinct = sorted(set(texts))
    vectors = rng.normal(size=(len(distinct), 32)).astype(np.float32)
    cache = CachedEmbeddings(inputs=encoder_inputs(TextEncoder.E5_SMALL, "title", distinct), embeddings=vectors,
                             meta={"features": ["title"]})
    monkeypatch.setattr(column_encoders, "cached_embeddings", lambda encoder, dataset, size, fold: cache)
    monkeypatch.setattr(e5_encoder, "get_vanilla_e5", lambda *a, **k: pytest.fail("loaded E5 for a cached column"))
    x = DataFrame({"title": texts})
    expected = vectors[[distinct.index(t) for t in texts]]
    train_idx, test_idx = rng.permutation(60)[:45], np.arange(60)[::4]
    encoders, train_embeddings = make_encoder(TextEncoder.E5_SMALL, device="cpu", dataset=TEXT).fit(x.iloc[train_idx],
                                                                                                    columns=["title"])
    np.testing.assert_array_equal(train_embeddings["title"], expected[train_idx])
    x_test = transform_columns(x.iloc[test_idx], encoders=encoders)
    np.testing.assert_allclose(x_test.to_numpy(), encoders["title"].transform(expected[test_idx]), rtol=1e-6)
    with pytest.raises(KeyError, match="no cached embedding"):
        transform_columns(DataFrame({"title": ["a text the cache never saw"]}), encoders=encoders)


def test_image_features_look_their_files_up_in_the_cache(monkeypatch):
    rng = np.random.default_rng(0)
    files = [f"images/{i % 40}.jpg" for i in range(60)]
    distinct = sorted(set(files))
    vectors = rng.normal(size=(len(distinct), 64)).astype(np.float32)
    cache = CachedEmbeddings(inputs=distinct, embeddings=vectors, meta={"features": ["photo"]})
    monkeypatch.setattr(column_encoders, "cached_embeddings", lambda encoder, dataset, size, fold: cache)
    monkeypatch.setattr(dino_encoder, "get_image_encoder", lambda *a, **k: pytest.fail("loaded DINO for cached images"))
    x = DataFrame({"photo": files})
    expected = vectors[[distinct.index(f) for f in files]]
    train_idx, test_idx = rng.permutation(60)[:45], np.arange(60)[::4]
    dino = make_encoder(ImageEncoder.DINO_SMALL, device="cpu", dataset=IMAGE_TEXT, image_folder="unused")
    encoders, train_embeddings = dino.fit(x.iloc[train_idx], columns=["photo"])
    np.testing.assert_array_equal(train_embeddings["photo"], expected[train_idx])
    x_test = transform_columns(x.iloc[test_idx], encoders=encoders)
    np.testing.assert_allclose(x_test.to_numpy(), encoders["photo"].transform(expected[test_idx]), rtol=1e-5)


def _tuned_cache(texts, columns):
    distinct = sorted(set(texts))
    vectors = np.random.default_rng(0).normal(size=(len(columns) * len(distinct), 32)).astype(np.float32)
    inputs = [i for col in columns for i in encoder_inputs(TextEncoder.E5_SMALL_TAR, col, distinct)]
    return CachedEmbeddings(inputs=inputs, embeddings=vectors, meta={"features": columns})


def test_a_tuned_run_reads_its_folds_embeddings(monkeypatch):
    texts = [f"t{i % 40}" for i in range(60)]
    asked = []

    def fold_cache(encoder, dataset, size, fold):
        asked.append((encoder, dataset, size, fold))
        return _tuned_cache(texts, ["title"])

    monkeypatch.setattr(column_encoders, "cached_embeddings", fold_cache)
    monkeypatch.setattr(e5_encoder, "finetune_e5", lambda *args, **kwargs: pytest.fail("tuned a cached fold"))
    e5 = make_encoder(TextEncoder.E5_SMALL_TAR, device="cpu", dataset=TEXT, size=Size.TEN_K, fold=2)
    encoders, _ = e5.fit(DataFrame({"title": texts}), columns=["title"])
    assert asked == [(TextEncoder.E5_SMALL_TAR, TEXT, Size.TEN_K, 2)]
    assert transform_columns(DataFrame({"title": ["t3"]}), encoders=encoders).shape == (1, 30)


def test_a_tuned_run_needs_its_fold():
    with pytest.raises(ValueError, match="per fold"):
        make_encoder(TextEncoder.E5_SMALL_TAR, device="cpu", dataset=TEXT).fit(DataFrame({"title": ["a"] * 40}),
                                                                               columns=["title"])


def test_a_fold_tuned_on_other_columns_raises(monkeypatch):
    texts = [f"t{i % 40}" for i in range(60)]
    monkeypatch.setattr(column_encoders, "cached_embeddings",
                        lambda encoder, dataset, size, fold: _tuned_cache(texts, ["title", "body"]))
    with pytest.raises(ValueError, match="tuned on"):
        e5 = make_encoder(TextEncoder.E5_SMALL_TAR, device="cpu", dataset=TEXT, size=Size.TEN_K, fold=0)
        e5.fit(DataFrame({"title": texts}), columns=["title"])


def test_embedding_jobs_write_a_jobs_file(tmp_path):
    from multabench.benchmark.sweep import write_jobs_file
    jobs = list_jobs(encoders=[TextEncoder.E5_SMALL_TAR], datasets=[TEXT], folds=[0])
    write_jobs_file(jobs, path=str(tmp_path / "jobs.txt"), python="python", output_dir="out")
    assert (tmp_path / "jobs.txt").read_text() == jobs[0].command(python="python", output_dir="out") + "\n"


def test_text_embeddings_are_standardized_before_pca():
    embeddings = np.random.default_rng(0).normal(size=(200, 40))
    embeddings[:, 0] *= 1000
    encoder = E5ColumnEncoder("title", model=None, tokenizer=None, device="cpu").fit(embeddings)
    first = encoder.transform(embeddings)[:, 0]
    assert abs(np.corrcoef(first, embeddings[:, 0])[0, 1]) < 0.5
