import argparse
import os
import subprocess
import sys

from multabench.benchmark.splits import SIZE_10K, SIZES, SPLITS
from multabench.benchmark.sweep import (DEFAULT_IMAGE_ENCODERS, DEFAULT_TEXT_ENCODERS, csv_arg, list_runs, pending_runs,
                                        run_local, sbatch_commands, write_jobs_file, write_slurm_script)
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.dino.constants import ImageEncoder
from multabench.e5.constants import TextEncoder

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark.py over models × datasets × encoders × folds. "
                                                 "Runs that already have a result JSON are left out.")
    parser.add_argument('--models', type=csv_arg(), required=True, help="model short names, e.g. light,cat,rf")
    parser.add_argument('--datasets', type=csv_arg([d.name for d in MulTaBenchDatasetID]), default=None,
                        help="dataset names; default: all")
    parser.add_argument('--text_encoders', type=csv_arg(list(TextEncoder), cast=TextEncoder), default=DEFAULT_TEXT_ENCODERS,
                        help="used on text datasets; image datasets with text columns always use e5-small")
    parser.add_argument('--image_encoders', type=csv_arg(list(ImageEncoder), cast=ImageEncoder), default=DEFAULT_IMAGE_ENCODERS,
                        help="used on datasets with images")
    parser.add_argument('--folds', type=csv_arg([str(f) for f in range(SPLITS)]), default=[str(f) for f in range(SPLITS)])
    parser.add_argument('--size', type=str, default=SIZE_10K, choices=SIZES)
    parser.add_argument('--output_dir', type=str, default='runs')
    parser.add_argument('--jobs_file', type=str, default='sweep_jobs.txt')
    parser.add_argument('--launcher', type=str, default=None, choices=["local", "slurm"],
                        help="default: only write the jobs file, one benchmark.py command per line")
    parser.add_argument('--sbatch', type=str, default="", help='extra sbatch arguments, given with "=", e.g. --sbatch="--partition=gpu --gres=gpu:1"')
    parser.add_argument('--max_parallel', type=int, default=None, help="Slurm array tasks running at once")
    parser.add_argument('--submit', action='store_true', default=False, help="run the sbatch commands instead of printing them")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    jobs_file = os.path.abspath(args.jobs_file)
    datasets = [MulTaBenchDatasetID[d] for d in args.datasets] if args.datasets else list(MulTaBenchDatasetID)
    runs = list_runs(models=args.models, datasets=datasets, text_encoders=args.text_encoders,
                     image_encoders=args.image_encoders, folds=[int(f) for f in args.folds], size=args.size)
    pending = pending_runs(runs, output_dir=output_dir)
    print(f"{len(runs)} runs, {len(runs) - len(pending)} already have a result, {len(pending)} to run")
    if not pending:
        sys.exit(0)
    write_jobs_file(pending, path=jobs_file, python=sys.executable, output_dir=output_dir)
    print(f"Wrote {jobs_file}", flush=True)

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
