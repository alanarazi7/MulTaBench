import numpy as np
import pytest
from pandas import DataFrame

from multabench.baselines.preprocessing.nulls import MISSING_VALUE
from multabench.datasets.all_datasets import MulTaBenchDatasetID, dataset_modality
from multabench.dino.constants import ImageEncoder
from multabench.e5.constants import TextEncoder, format_e5_passage
from multabench.embeddings.hub import (CACHED_ENCODERS, META_JSON, embeddings_dir, embeddings_repo_id, embeds_dataset,
                                       encoder_inputs, input_values, parse_encoder, read_embeddings, write_embeddings)
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
