import os
import shlex
from dataclasses import dataclass
from os.path import join
from typing import Iterable, List, Optional

from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.embeddings.hub import META_JSON, CachedEncoder, embeddings_dir, embeds_dataset, is_tuned, repo_name


@dataclass(frozen=True)
class EmbeddingJob:
    """One embed.py run: a frozen encoder over a whole dataset, or one fine-tune for a (size, fold) split."""
    encoder: CachedEncoder
    dataset: MulTaBenchDatasetID
    size: Optional[str] = None
    fold: Optional[int] = None

    @property
    def path(self) -> str:
        return join(repo_name(self.encoder), embeddings_dir(self.dataset, self.encoder, size=self.size, fold=self.fold))

    def command(self, python: str, output_dir: str) -> str:
        args = [python, "embed.py", "--encoder", self.encoder, "--dataset_name", self.dataset.name,
                "--output_dir", output_dir]
        if is_tuned(self.encoder):
            args += ["--size", self.size, "--fold", str(self.fold)]
        return shlex.join(args)


def list_jobs(encoders: Iterable[CachedEncoder], datasets: Iterable[MulTaBenchDatasetID], folds: Iterable[int],
              size: str) -> List[EmbeddingJob]:
    jobs = []
    for encoder in encoders:
        for dataset in datasets:
            if not embeds_dataset(encoder, dataset):
                continue
            if is_tuned(encoder):
                jobs += [EmbeddingJob(encoder, dataset, size=size, fold=fold) for fold in folds]
            else:
                jobs.append(EmbeddingJob(encoder, dataset))
    return jobs


def pending_jobs(jobs: List[EmbeddingJob], output_dir: str) -> List[EmbeddingJob]:
    return [job for job in jobs if not os.path.exists(join(output_dir, job.path, META_JSON))]
