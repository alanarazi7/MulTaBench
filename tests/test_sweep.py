from multabench.benchmark.runs import Run
from multabench.benchmark.sweep import (DEFAULT_IMAGE_ENCODERS, DEFAULT_TEXT_ENCODERS, IMAGE_TEXT_TEXT_ENCODER,
                                        encoder_pairs, list_runs, pending_runs, sbatch_commands)
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID
from multabench.e5.constants import TextEncoder

TEXT = MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING


def test_each_modality_gets_only_its_encoders():
    text = encoder_pairs(Modality.TEXT, DEFAULT_TEXT_ENCODERS, DEFAULT_IMAGE_ENCODERS)
    image = encoder_pairs(Modality.IMAGE, DEFAULT_TEXT_ENCODERS, DEFAULT_IMAGE_ENCODERS)
    image_text = encoder_pairs(Modality.IMAGE_TEXT, DEFAULT_TEXT_ENCODERS, DEFAULT_IMAGE_ENCODERS)
    assert text == [(t, None) for t in DEFAULT_TEXT_ENCODERS]
    assert image == [(None, i) for i in DEFAULT_IMAGE_ENCODERS]
    assert image_text == [(IMAGE_TEXT_TEXT_ENCODER, i) for i in DEFAULT_IMAGE_ENCODERS]


def test_runs_cover_encoders_and_folds():
    runs = list_runs(models=["light"], datasets=[TEXT], text_encoders=DEFAULT_TEXT_ENCODERS,
                     image_encoders=DEFAULT_IMAGE_ENCODERS, folds=[0, 1], size="10k")
    assert len(runs) == len(DEFAULT_TEXT_ENCODERS) * 2
    assert {run.image_encoder for run in runs} == {None}


def test_runs_with_a_result_are_not_pending(tmp_path):
    done = Run("light", TEXT.name, TextEncoder.TFIDF, None, "10k", 0)
    todo = Run("light", TEXT.name, TextEncoder.TFIDF, None, "10k", 1)
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
