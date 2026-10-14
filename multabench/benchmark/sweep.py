"""Lists the benchmark.py runs of a sweep and runs them, locally or as a Slurm array."""
import argparse
import os
import shlex
import subprocess
from typing import Iterable, List, Optional, Set, Tuple

from multabench.benchmark.load import download_dataset
from multabench.benchmark.runs import Run
from multabench.benchmark.splits import Size
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.dino.constants import IMAGE_ENCODERS, ImageEncoder
from multabench.e5.constants import TEXT_ENCODERS, TextEncoder
from multabench.embeddings.hub import cached_encoder, download_embeddings, reads_cache

SLURM_ARRAY_LIMIT = 1000
DEFAULT_TEXT_ENCODERS = [TextEncoder.TFIDF, TextEncoder.E5_SMALL, TextEncoder.E5_SMALL_TAR]
DEFAULT_IMAGE_ENCODERS = [ImageEncoder.DINO_SMALL, ImageEncoder.DINO_SMALL_TAR]
# Image datasets with text columns vary only the image encoder; their text is embedded with frozen E5.
IMAGE_TEXT_TEXT_ENCODER = TextEncoder.E5_SMALL


def select_datasets(names: Optional[List[str]], modalities: Optional[List[Modality]]) -> List[MulTaBenchDatasetID]:
    datasets = [MulTaBenchDatasetID[d] for d in names] if names else list(MulTaBenchDatasetID)
    return [d for d in datasets if not modalities or dataset_modality(d) in modalities]


def encoder_pairs(modality: Modality, text_encoders: List[TextEncoder], image_encoders: List[ImageEncoder],
                  embeds_text: bool = True) -> List[Tuple[Optional[TextEncoder], Optional[ImageEncoder]]]:
    """A model that reads raw text gets no text encoder, so it runs once per dataset rather than once per encoder."""
    if modality == Modality.TEXT:
        return [(text_encoder, None) for text_encoder in text_encoders] if embeds_text else [(None, None)]
    if modality == Modality.IMAGE:
        return [(None, image_encoder) for image_encoder in image_encoders]
    text_encoder = IMAGE_TEXT_TEXT_ENCODER if embeds_text else None
    return [(text_encoder, image_encoder) for image_encoder in image_encoders]


def list_runs(models: Iterable[str], datasets: Iterable[MulTaBenchDatasetID], text_encoders: List[TextEncoder],
              image_encoders: List[ImageEncoder], folds: Iterable[int], size: Size,
              raw_text_models: Set[str] = frozenset()) -> List[Run]:
    runs = []
    for model in models:
        for dataset in datasets:
            pairs = encoder_pairs(dataset_modality(dataset), text_encoders, image_encoders,
                                  embeds_text=model not in raw_text_models)
            for text_encoder, image_encoder in pairs:
                for fold in folds:
                    runs.append(Run(model, dataset.name, text_encoder, image_encoder, size, fold))
    return runs


def csv_arg(choices=None, cast=str):
    def parse(value: str):
        items = [v for v in value.split(",") if v]
        unknown = [v for v in items if choices is not None and v not in choices]
        if unknown:
            raise argparse.ArgumentTypeError(f"unknown {unknown}; choose from {sorted(map(str, choices))}")
        return [cast(v) for v in items]
    return parse


def pending_runs(runs: List[Run], output_dir: str) -> List[Run]:
    return [run for run in runs if not os.path.exists(os.path.join(output_dir, f"{run.name}.json"))]


def prefetch(runs: List[Run]):
    """Downloads the runs' datasets and cached embeddings once, before they start: runs that download the same files
    at once into a Hugging Face cache on NFS fail with stale file handles."""
    for dataset in sorted({run.dataset for run in runs}):
        download_dataset(MulTaBenchDatasetID[dataset])
    pairs = {(TEXT_ENCODERS[run.text_encoder], run.dataset) for run in runs if run.text_encoder}
    pairs |= {(IMAGE_ENCODERS[run.image_encoder], run.dataset) for run in runs if run.image_encoder}
    for encoder, dataset in sorted(pairs):
        if reads_cache(encoder):
            download_embeddings(cached_encoder(encoder.encoder_name, encoder.tune_encoder), MulTaBenchDatasetID[dataset])


def write_jobs_file(runs: List[Run], path: str, python: str, output_dir: str, official: bool = False):
    with open(path, "w") as f:
        for run in runs:
            f.write(run.command(python=python, output_dir=output_dir, official=official) + "\n")


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
# Slurm may give the job a whole machine; srun limits the run to the CPUs the job asked for.
eval "srun ${{SLURM_CPUS_PER_TASK:+--cpus-per-task=$SLURM_CPUS_PER_TASK}} $COMMAND"
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
