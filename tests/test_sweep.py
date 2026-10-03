from multabench.benchmark.sweep import Job, list_jobs, pending_jobs, sbatch_commands
from multabench.datasets.all_datasets import MulTaBenchDatasetID

IMAGE = MulTaBenchDatasetID.MUL_IMAGE_PETFINDER
TEXT = MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING


def test_text_datasets_run_with_the_first_image_encoder_only():
    jobs = list_jobs(models=["light"], datasets=[IMAGE, TEXT], text_encoders=["tfidf", "e5-small"],
                     image_encoders=["dino-small", "dino-small-tar"], folds=[0, 1], size="10k")
    assert sum(job.dataset == IMAGE.name for job in jobs) == 2 * 2 * 2
    assert {job.image_encoder for job in jobs if job.dataset == TEXT.name} == {"dino-small"}
    assert sum(job.dataset == TEXT.name for job in jobs) == 2 * 2


def test_jobs_with_a_result_are_not_pending(tmp_path):
    done, todo = Job("light", TEXT.name, "tfidf", "dino-small", "10k", 0), Job("light", TEXT.name, "tfidf", "dino-small", "10k", 1)
    open(done.result_path(str(tmp_path)), "w").close()
    assert pending_jobs([done, todo], output_dir=str(tmp_path)) == [todo]


def test_sbatch_commands_split_large_arrays():
    commands = sbatch_commands(2500, script="s.sbatch", sbatch_args="--gres=gpu:1", max_parallel=50)
    assert commands == [
        "sbatch --array=0-999%50 --export=ALL,OFFSET=0 --gres=gpu:1 s.sbatch",
        "sbatch --array=0-999%50 --export=ALL,OFFSET=1000 --gres=gpu:1 s.sbatch",
        "sbatch --array=0-499%50 --export=ALL,OFFSET=2000 --gres=gpu:1 s.sbatch",
    ]
