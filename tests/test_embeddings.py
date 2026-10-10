import numpy as np
import pytest

from multabench.datasets.all_datasets import MulTaBenchDatasetID, dataset_modality
from multabench.dino.constants import ImageEncoder
from multabench.e5.constants import TextEncoder
from multabench.embeddings.hub import (CACHED_ENCODERS, META_JSON, embeddings_dir, embeddings_repo_id, embeds_dataset,
                                       parse_encoder, read_embeddings, select_rows, write_embeddings)
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
