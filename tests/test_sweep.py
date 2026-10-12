from multabench.benchmark.runs import Run
from multabench.benchmark.splits import Size
from multabench.benchmark import sweep
from multabench.benchmark.sweep import (DEFAULT_IMAGE_ENCODERS, DEFAULT_TEXT_ENCODERS, IMAGE_TEXT_TEXT_ENCODER,
                                        encoder_pairs, list_runs, pending_runs, prefetch, sbatch_commands,
                                        select_datasets)
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.dino.constants import ImageEncoder
from multabench.e5.constants import TextEncoder

TEXT = MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING
IMAGE_TEXT = MulTaBenchDatasetID.MUL_IMAGE_PETFINDER


def test_each_modality_gets_only_its_encoders():
    text = encoder_pairs(Modality.TEXT, DEFAULT_TEXT_ENCODERS, DEFAULT_IMAGE_ENCODERS)
    image = encoder_pairs(Modality.IMAGE, DEFAULT_TEXT_ENCODERS, DEFAULT_IMAGE_ENCODERS)
    image_text = encoder_pairs(Modality.IMAGE_TEXT, DEFAULT_TEXT_ENCODERS, DEFAULT_IMAGE_ENCODERS)
    assert text == [(t, None) for t in DEFAULT_TEXT_ENCODERS]
    assert image == [(None, i) for i in DEFAULT_IMAGE_ENCODERS]
    assert image_text == [(IMAGE_TEXT_TEXT_ENCODER, i) for i in DEFAULT_IMAGE_ENCODERS]


def test_runs_cover_encoders_and_folds():
    runs = list_runs(models=["light"], datasets=[TEXT], text_encoders=DEFAULT_TEXT_ENCODERS,
                     image_encoders=DEFAULT_IMAGE_ENCODERS, folds=[0, 1], size=Size.TEN_K)
    assert len(runs) == len(DEFAULT_TEXT_ENCODERS) * 2
    assert {run.image_encoder for run in runs} == {None}


def test_runs_with_a_result_are_not_pending(tmp_path):
    done = Run("light", TEXT.name, TextEncoder.TFIDF, None, Size.TEN_K, 0)
    todo = Run("light", TEXT.name, TextEncoder.TFIDF, None, Size.TEN_K, 1)
    assert done.name == "light_BIN_TEXT_FAKE_JOB_POSTING_tfidf_none_10k_0"
    open(tmp_path / f"{done.name}.json", "w").close()
    assert pending_runs([done, todo], output_dir=str(tmp_path)) == [todo]


def test_sbatch_commands_split_large_arrays():
    commands = sbatch_commands(2500, script="s.sbatch", sbatch_args="--gres=gpu:1", max_parallel=50)
    assert commands == [
        "sbatch --array=0-999%50 --export=ALL,OFFSET=0 --gres=gpu:1 s.sbatch",
        "sbatch --array=0-999%50 --export=ALL,OFFSET=1000 --gres=gpu:1 s.sbatch",
        "sbatch --array=0-499%50 --export=ALL,OFFSET=2000 --gres=gpu:1 s.sbatch",
    ]


def test_select_datasets_by_modality():
    text = select_datasets(None, [Modality.TEXT])
    assert len(text) == 40 and all(dataset_modality(d) == Modality.TEXT for d in text)
    assert select_datasets([TEXT.name, "MUL_IMAGE_PETFINDER"], [Modality.TEXT]) == [TEXT]
    assert select_datasets(None, None) == list(MulTaBenchDatasetID)


def test_prefetch_downloads_each_dataset_and_its_cached_embeddings_once(monkeypatch):
    datasets, embeddings = [], []
    monkeypatch.setattr(sweep, "download_dataset", datasets.append)
    monkeypatch.setattr(sweep, "download_embeddings", lambda encoder, dataset: embeddings.append((encoder, dataset)))
    runs = list_runs(models=["light", "cat"], datasets=[TEXT], text_encoders=DEFAULT_TEXT_ENCODERS,
                     image_encoders=DEFAULT_IMAGE_ENCODERS, folds=[0, 1], size=Size.TEN_K)
    runs += list_runs(models=["light"], datasets=[IMAGE_TEXT], text_encoders=DEFAULT_TEXT_ENCODERS,
                      image_encoders=DEFAULT_IMAGE_ENCODERS, folds=[0], size=Size.TEN_K)
    prefetch(runs)
    assert len(datasets) == 2 and set(datasets) == {TEXT, IMAGE_TEXT}
    assert len(embeddings) == 3 and set(embeddings) == {(TextEncoder.E5_SMALL, TEXT), (TextEncoder.E5_SMALL, IMAGE_TEXT),
                                                        (ImageEncoder.DINO_SMALL, IMAGE_TEXT)}
