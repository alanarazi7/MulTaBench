from multabench.benchmark.runs import Run
from multabench.benchmark.sweep import encoder_pairs, list_runs, pending_runs, sbatch_commands
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID

TEXT = MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING
TEXT_ENCODERS = ["tfidf", "e5-small"]
IMAGE_ENCODERS = ["dino-small", "dino-small-tar"]


def test_each_modality_gets_only_its_encoders():
    assert encoder_pairs(Modality.TEXT, TEXT_ENCODERS, IMAGE_ENCODERS) == [("tfidf", None), ("e5-small", None)]
    assert encoder_pairs(Modality.IMAGE, TEXT_ENCODERS, IMAGE_ENCODERS) == [(None, "dino-small"), (None, "dino-small-tar")]
    assert encoder_pairs(Modality.IMAGE_TEXT, TEXT_ENCODERS, IMAGE_ENCODERS) == [("e5-small", "dino-small"),
                                                                                ("e5-small", "dino-small-tar")]


def test_runs_cover_encoders_and_folds():
    runs = list_runs(models=["light"], datasets=[TEXT], text_encoders=TEXT_ENCODERS, image_encoders=IMAGE_ENCODERS,
                     folds=[0, 1], size="10k")
    assert len(runs) == 2 * 2
    assert {run.image_encoder for run in runs} == {None}


def test_runs_with_a_result_are_not_pending(tmp_path):
    done, todo = Run("light", TEXT.name, "tfidf", None, "10k", 0), Run("light", TEXT.name, "tfidf", None, "10k", 1)
    open(tmp_path / f"{done.name}.json", "w").close()
    assert pending_runs([done, todo], output_dir=str(tmp_path)) == [todo]


def test_sbatch_commands_split_large_arrays():
    commands = sbatch_commands(2500, script="s.sbatch", sbatch_args="--gres=gpu:1", max_parallel=50)
    assert commands == [
        "sbatch --array=0-999%50 --export=ALL,OFFSET=0 --gres=gpu:1 s.sbatch",
        "sbatch --array=0-999%50 --export=ALL,OFFSET=1000 --gres=gpu:1 s.sbatch",
        "sbatch --array=0-499%50 --export=ALL,OFFSET=2000 --gres=gpu:1 s.sbatch",
    ]
