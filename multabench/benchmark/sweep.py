"""Lists the benchmark.py runs of a sweep and runs them, locally or as a Slurm array."""
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Iterable, List

from multabench.benchmark.runs import result_path
from multabench.datasets.all_datasets import MulTaBenchDatasetID, is_image_dataset

SLURM_ARRAY_LIMIT = 1000


@dataclass(frozen=True)
class Job:
    model: str
    dataset: str
    text_encoder: str
    image_encoder: str
    size: str
    fold: int

    def result_path(self, output_dir: str) -> str:
        return result_path(output_dir, self.model, self.dataset, self.text_encoder, self.image_encoder, self.size, self.fold)

    def command(self, python: str, output_dir: str) -> str:
        args = [python, "benchmark.py", "--model", self.model, "--dataset_name", self.dataset,
                "--text_encoder", self.text_encoder, "--image_encoder", self.image_encoder,
                "--size", self.size, "--fold", str(self.fold), "--output_dir", output_dir]
        return shlex.join(args)


def list_jobs(models: Iterable[str], datasets: Iterable[MulTaBenchDatasetID], text_encoders: List[str],
              image_encoders: List[str], folds: Iterable[int], size: str) -> List[Job]:
    """Text datasets have no images, so they run with the first image encoder only."""
    jobs = []
    for model in models:
        for dataset in datasets:
            dataset_image_encoders = image_encoders if is_image_dataset(dataset) else image_encoders[:1]
            for text_encoder in text_encoders:
                for image_encoder in dataset_image_encoders:
                    for fold in folds:
                        jobs.append(Job(model, dataset.name, text_encoder, image_encoder, size, fold))
    return jobs


def pending_jobs(jobs: List[Job], output_dir: str) -> List[Job]:
    return [job for job in jobs if not os.path.exists(job.result_path(output_dir))]


def write_jobs_file(jobs: List[Job], path: str, python: str, output_dir: str):
    with open(path, "w") as f:
        for job in jobs:
            f.write(job.command(python=python, output_dir=output_dir) + "\n")


def run_local(jobs_file: str, repo_dir: str) -> int:
    """Runs each line in its own process, one after another; a failed run doesn't stop the others."""
    with open(jobs_file) as f:
        commands = [line.strip() for line in f if line.strip()]
    failed = 0
    for i, command in enumerate(commands, start=1):
        print(f"[{i}/{len(commands)}] {command}", flush=True)
        failed += subprocess.run(shlex.split(command), cwd=repo_dir).returncode != 0
    print(f"Done: {len(commands) - failed} succeeded, {failed} failed")
    return failed


def write_slurm_script(path: str, jobs_file: str, repo_dir: str, log_dir: str):
    """Array task i runs line OFFSET + i + 1 of the jobs file."""
    script = f"""#!/bin/bash
#SBATCH --job-name=multabench
#SBATCH --output={log_dir}/%A_%a.out
set -euo pipefail
cd {shlex.quote(repo_dir)}
LINE=$(( ${{OFFSET:-0}} + SLURM_ARRAY_TASK_ID + 1 ))
COMMAND=$(sed -n "${{LINE}}p" {shlex.quote(jobs_file)})
echo "$COMMAND"
eval "$COMMAND"
"""
    with open(path, "w") as f:
        f.write(script)


def sbatch_commands(n_jobs: int, script: str, sbatch_args: str, max_parallel: int | None) -> List[str]:
    """One array per SLURM_ARRAY_LIMIT jobs, since clusters cap the array size."""
    commands = []
    for offset in range(0, n_jobs, SLURM_ARRAY_LIMIT):
        last = min(SLURM_ARRAY_LIMIT, n_jobs - offset) - 1
        array = f"0-{last}" + (f"%{max_parallel}" if max_parallel else "")
        parts = ["sbatch", f"--array={array}", f"--export=ALL,OFFSET={offset}", sbatch_args, shlex.quote(script)]
        commands.append(" ".join(p for p in parts if p))
    return commands
