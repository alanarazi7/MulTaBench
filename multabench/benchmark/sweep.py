"""Lists the benchmark.py runs of a sweep and runs them, locally or as a Slurm array."""
import os
import shlex
import subprocess
from typing import Iterable, List, Optional, Tuple

from multabench.benchmark.runs import Run
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality

SLURM_ARRAY_LIMIT = 1000
# Image datasets with text columns vary only the image encoder; their text is embedded with frozen E5.
IMAGE_TEXT_TEXT_ENCODER = "e5-small"


def encoder_pairs(modality: Modality, text_encoders: List[str],
                  image_encoders: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    if modality == Modality.TEXT:
        return [(text_encoder, None) for text_encoder in text_encoders]
    if modality == Modality.IMAGE:
        return [(None, image_encoder) for image_encoder in image_encoders]
    return [(IMAGE_TEXT_TEXT_ENCODER, image_encoder) for image_encoder in image_encoders]


def list_runs(models: Iterable[str], datasets: Iterable[MulTaBenchDatasetID], text_encoders: List[str],
              image_encoders: List[str], folds: Iterable[int], size: str) -> List[Run]:
    runs = []
    for model in models:
        for dataset in datasets:
            for text_encoder, image_encoder in encoder_pairs(dataset_modality(dataset), text_encoders, image_encoders):
                for fold in folds:
                    runs.append(Run(model, dataset.name, text_encoder, image_encoder, size, fold))
    return runs


def pending_runs(runs: List[Run], output_dir: str) -> List[Run]:
    return [run for run in runs if not os.path.exists(os.path.join(output_dir, f"{run.name}.json"))]


def write_jobs_file(runs: List[Run], path: str, python: str, output_dir: str):
    with open(path, "w") as f:
        for run in runs:
            f.write(run.command(python=python, output_dir=output_dir) + "\n")


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
