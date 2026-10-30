import json
import os

from multabench.benchmark.runs import Run
from multabench.benchmark.splits import Size
from multabench.benchmark import sweep
from multabench.benchmark.sweep import (DEFAULT_IMAGE_ENCODERS, DEFAULT_TEXT_ENCODERS,
                                        encoder_pairs, list_runs, pending_runs, prefetch, sbatch_commands,
                                        select_datasets, venv_python, write_jobs_file)
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
    assert image_text == [(TextEncoder.E5_SMALL, i) for i in DEFAULT_IMAGE_ENCODERS]


def test_large_image_runs_embed_text_with_large_e5():
    assert encoder_pairs(Modality.IMAGE_TEXT, [TextEncoder.E5_LARGE], [ImageEncoder.DINO_LARGE,
                                                                        ImageEncoder.DINO_LARGE_TAR]) == [
        (TextEncoder.E5_LARGE, ImageEncoder.DINO_LARGE), (TextEncoder.E5_LARGE, ImageEncoder.DINO_LARGE_TAR)]


def test_raw_text_models_get_no_text_encoder():
    text = encoder_pairs(Modality.TEXT, DEFAULT_TEXT_ENCODERS, DEFAULT_IMAGE_ENCODERS, embeds_text=False)
    image_text = encoder_pairs(Modality.IMAGE_TEXT, DEFAULT_TEXT_ENCODERS, DEFAULT_IMAGE_ENCODERS, embeds_text=False)
    assert text == [(None, None)]
    assert image_text == [(None, i) for i in DEFAULT_IMAGE_ENCODERS]


def test_raw_text_models_run_once_per_dataset_and_fold():
    runs = list_runs(models=["ctx", "light"], datasets=[TEXT], text_encoders=DEFAULT_TEXT_ENCODERS,
                     image_encoders=DEFAULT_IMAGE_ENCODERS, folds=[0, 1], size=Size.TEN_K, raw_text_models={"ctx"})
    ctx = [run for run in runs if run.model == "ctx"]
    assert [run.name for run in ctx] == ["ctx_BIN_TEXT_FAKE_JOB_POSTING_none_none_10k_0",
                                         "ctx_BIN_TEXT_FAKE_JOB_POSTING_none_none_10k_1"]
    assert "--text_encoder" not in ctx[0].command(python="python", output_dir="runs")
    assert len(runs) - len(ctx) == len(DEFAULT_TEXT_ENCODERS) * 2


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


def test_failed_runs_rerun_only_when_asked(tmp_path):
    ok, failed = (Run("light", TEXT.name, TextEncoder.TFIDF, None, Size.TEN_K, fold) for fold in (0, 1))
    for run, status in ((ok, "ok"), (failed, "error")):
        (tmp_path / f"{run.name}.json").write_text(json.dumps({"status": status}))
    assert pending_runs([ok, failed], output_dir=str(tmp_path)) == []
    assert pending_runs([ok, failed], output_dir=str(tmp_path), rerun_failed=True) == [failed]
    jobs = tmp_path / "jobs.txt"
    write_jobs_file([failed], path=str(jobs), python="python", output_dir="runs", overwrite=True)
    assert jobs.read_text().strip().endswith("--overwrite --text_encoder tfidf")


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
    monkeypatch.setattr(sweep, "download_embeddings",
                        lambda encoder, dataset, size, fold: embeddings.append((encoder, dataset, size, fold)))
    runs = list_runs(models=["light", "cat"], datasets=[TEXT], text_encoders=DEFAULT_TEXT_ENCODERS,
                     image_encoders=DEFAULT_IMAGE_ENCODERS, folds=[0, 1], size=Size.TEN_K)
    runs += list_runs(models=["light"], datasets=[IMAGE_TEXT], text_encoders=DEFAULT_TEXT_ENCODERS,
                      image_encoders=DEFAULT_IMAGE_ENCODERS, folds=[0], size=Size.TEN_K)
    prefetch(runs)
    assert len(datasets) == 2 and set(datasets) == {TEXT, IMAGE_TEXT}
    tuned = [(TextEncoder.E5_SMALL_TAR, TEXT, Size.TEN_K, 0), (TextEncoder.E5_SMALL_TAR, TEXT, Size.TEN_K, 1),
             (ImageEncoder.DINO_SMALL_TAR, IMAGE_TEXT, Size.TEN_K, 0)]
    frozen = [(TextEncoder.E5_SMALL, TEXT, None, None), (TextEncoder.E5_SMALL, IMAGE_TEXT, None, None),
              (ImageEncoder.DINO_SMALL, IMAGE_TEXT, None, None)]
    assert sorted(embeddings, key=str) == sorted(tuned + frozen, key=str)


def test_official_runs_check_their_hardware():
    run = Run("light", TEXT.name, TextEncoder.E5_SMALL, None, Size.TEN_K, 0)
    assert "--official" in run.command(python="python", output_dir="runs", official=True)
    assert "--official" not in run.command(python="python", output_dir="runs")


def test_a_model_with_its_own_venv_runs_with_its_python(tmp_path):
    runs = [Run("light", TEXT.name, TextEncoder.E5_SMALL, None, Size.TEN_K, 0),
            Run("causilo", TEXT.name, TextEncoder.E5_SMALL, None, Size.TEN_K, 0)]
    jobs = tmp_path / "jobs.txt"
    write_jobs_file(runs, path=str(jobs), python="shared/python", output_dir="runs",
                    model_pythons={"causilo": venv_python("repo", "causilo")})
    lines = jobs.read_text().splitlines()
    assert lines[0].startswith("shared/python benchmark.py --model light ")
    assert lines[1].startswith("repo/.venv-causilo/bin/python benchmark.py --model causilo ")


def test_causilo_declares_its_venv():
    from multabench.baselines.causilo import Causilo
    from multabench.baselines.lgbm import LightGBM
    assert Causilo.ENV == "causilo" and LightGBM.ENV is None


def test_every_model_venv_has_a_lock():
    from benchmark import BASELINES
    repo = os.path.join(os.path.dirname(__file__), "..")
    envs = {m.ENV for m in BASELINES if m.ENV}
    missing = [env for env in envs if not os.path.exists(os.path.join(repo, f"requirements-{env}.txt"))]
    assert not missing
