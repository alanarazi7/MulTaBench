import argparse
import os
import subprocess
import sys
import time

from benchmark import BASELINES, SHORT2MODELS
from multabench.benchmark.splits import DEFAULT_SIZE, SPLITS, Size
from multabench.benchmark.sweep import (DEFAULT_IMAGE_ENCODERS, DEFAULT_TEXT_ENCODERS, csv_arg, list_runs, pending_runs, prefetch,
                                        run_local, sbatch_commands, select_datasets, venv_python, write_jobs_file,
                                        write_slurm_script)
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID
from multabench.dino.constants import ImageEncoder
from multabench.e5.constants import TextEncoder

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark.py over models × datasets × encoders × folds. "
                                                 "Runs that already have a result JSON are left out.")
    parser.add_argument('--models', type=csv_arg(), required=True, help="model short names, e.g. light,cat,rf")
    parser.add_argument('--datasets', type=csv_arg([d.name for d in MulTaBenchDatasetID]), default=None,
                        help="dataset names; default: all")
    parser.add_argument('--modalities', type=csv_arg([m.value for m in Modality], cast=Modality), default=None,
                        help="keep only the datasets of these modalities: text, image, image-text; default: all")
    parser.add_argument('--text_encoders', type=csv_arg(list(TextEncoder), cast=TextEncoder), default=DEFAULT_TEXT_ENCODERS,
                        help="used on text datasets; image datasets use E5 of the image encoder's size, raw-text models none")
    parser.add_argument('--image_encoders', type=csv_arg(list(ImageEncoder), cast=ImageEncoder), default=DEFAULT_IMAGE_ENCODERS,
                        help="used on datasets with images")
    parser.add_argument('--folds', type=csv_arg([str(f) for f in range(SPLITS)]), default=[str(f) for f in range(SPLITS)])
    parser.add_argument('--size', type=Size, default=DEFAULT_SIZE, choices=list(Size))
    parser.add_argument('--output_dir', type=str, default='runs')
    parser.add_argument('--official', action='store_true', default=False, help="pass --official to every run")
    parser.add_argument('--rerun_failed', action='store_true', default=False,
                        help="also rerun the runs whose result is an error, e.g. after a rate limit or a missing token")
    parser.add_argument('--jobs_file', type=str, default=None,
                        help="default: sweep_jobs_<timestamp>.txt, so a new sweep never rewrites the file a queued "
                             "Slurm array reads its commands from")
    parser.add_argument('--launcher', type=str, default=None, choices=["local", "slurm"],
                        help="default: only write the jobs file, one benchmark.py command per line")
    parser.add_argument('--sbatch', type=str, default="", help='extra sbatch arguments, given with "=", e.g. --sbatch="--partition=gpu --gres=gpu:1"')
    parser.add_argument('--max_parallel', type=int, default=None, help="Slurm array tasks running at once")
    parser.add_argument('--submit', action='store_true', default=False, help="run the sbatch commands instead of printing them")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    jobs_file = os.path.abspath(args.jobs_file or time.strftime("sweep_jobs_%Y%m%d_%H%M%S.txt"))
    datasets = select_datasets(args.datasets, args.modalities)
    runs = list_runs(models=args.models, datasets=datasets, text_encoders=args.text_encoders,
                     image_encoders=args.image_encoders, folds=[int(f) for f in args.folds], size=args.size,
                     raw_text_models={m.SHORT_NAME for m in BASELINES if not m.USE_TEXT_EMBEDDINGS})
    pending = pending_runs(runs, output_dir=output_dir, rerun_failed=args.rerun_failed)
    print(f"{len(runs)} runs, {len(runs) - len(pending)} already have a result, {len(pending)} to run")
    if not pending:
        sys.exit(0)
    if args.official and len({SHORT2MODELS[run.model].NEEDS_GPU for run in pending}) > 1:
        parser.error("official GPU and CPU runs need different machines, so sweep their models apart (docs/running.md)")
    # Models with their own venv run with its python; the others with this one's.
    model_pythons = {m.SHORT_NAME: venv_python(REPO_DIR, m.ENV) for m in BASELINES if m.ENV and m.SHORT_NAME in args.models}
    missing = {model: python for model, python in model_pythons.items() if not os.path.exists(python)}
    if missing:
        parser.error(f"No venv for {sorted(missing)}: set it up with `source init.sh <ENV>` ({missing})")
    write_jobs_file(pending, path=jobs_file, python=sys.executable, output_dir=output_dir, official=args.official,
                    model_pythons=model_pythons, overwrite=args.rerun_failed)
    print(f"Wrote {jobs_file}", flush=True)

    if args.launcher:
        prefetch(pending)
    if args.launcher == "local":
        sys.exit(1 if run_local(jobs_file, repo_dir=REPO_DIR) else 0)
    if args.launcher == "slurm":
        log_dir = os.path.join(output_dir, "slurm_logs")
        os.makedirs(log_dir, exist_ok=True)
        script = os.path.splitext(jobs_file)[0] + ".sbatch"
        write_slurm_script(script, jobs_file=jobs_file, repo_dir=REPO_DIR, log_dir=log_dir)
        for command in sbatch_commands(len(pending), script=script, sbatch_args=args.sbatch, max_parallel=args.max_parallel):
            print(command, flush=True)
            if args.submit:
                subprocess.run(command, shell=True, check=True)
