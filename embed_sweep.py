import argparse
import os
import subprocess
import sys

from huggingface_hub import HfApi

from multabench.benchmark.sweep import csv_arg, run_local, sbatch_commands, write_jobs_file, write_slurm_script
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.embeddings.hub import CACHED_ENCODERS, embeddings_repo_id, parse_encoder, repo_name
from multabench.embeddings.jobs import list_jobs, pending_jobs

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def upload(output_dir: str, encoders):
    api = HfApi()
    for encoder in encoders:
        folder = os.path.join(output_dir, repo_name(encoder))
        if not os.path.isdir(folder):
            print(f"Nothing to upload for {encoder}: {folder} doesn't exist")
            continue
        repo_id = embeddings_repo_id(encoder)
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        api.upload_large_folder(repo_id=repo_id, folder_path=folder, repo_type="dataset")
        print(f"Uploaded {folder} to {repo_id}: {api.repo_info(repo_id, repo_type='dataset').sha}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run embed.py over encoders × datasets, then upload the embeddings to "
                                                 "Hugging Face. Jobs that already have their embeddings are left out.")
    parser.add_argument('--encoders', type=csv_arg(list(CACHED_ENCODERS), cast=parse_encoder), default=list(CACHED_ENCODERS))
    parser.add_argument('--datasets', type=csv_arg([d.name for d in MulTaBenchDatasetID]), default=None,
                        help="dataset names; default: every dataset the encoder applies to")
    parser.add_argument('--output_dir', type=str, default='embeddings')
    parser.add_argument('--jobs_file', type=str, default='embed_jobs.txt')
    parser.add_argument('--launcher', type=str, default=None, choices=["local", "slurm"],
                        help="default: only write the jobs file, one embed.py command per line")
    parser.add_argument('--sbatch', type=str, default="", help='extra sbatch arguments, given with "=", e.g. --sbatch="--partition=gpu --gres=gpu:1"')
    parser.add_argument('--max_parallel', type=int, default=None, help="Slurm array tasks running at once")
    parser.add_argument('--submit', action='store_true', default=False, help="run the sbatch commands instead of printing them")
    parser.add_argument('--upload', action='store_true', default=False,
                        help="upload --output_dir to one Hugging Face dataset repo per encoder, instead of computing")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if args.upload:
        upload(output_dir, args.encoders)
        sys.exit(0)
    jobs_file = os.path.abspath(args.jobs_file)
    datasets = [MulTaBenchDatasetID[d] for d in args.datasets] if args.datasets else list(MulTaBenchDatasetID)
    jobs = list_jobs(encoders=args.encoders, datasets=datasets)
    pending = pending_jobs(jobs, output_dir=output_dir)
    print(f"{len(jobs)} jobs, {len(jobs) - len(pending)} already have their embeddings, {len(pending)} to run")
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
