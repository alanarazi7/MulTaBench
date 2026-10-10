import os
import shlex
from dataclasses import dataclass
from os.path import join
from typing import Iterable, List

from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.embeddings.hub import META_JSON, CachedEncoder, embeddings_dir, embeds_dataset, repo_name


@dataclass(frozen=True)
class EmbeddingJob:
    """One embed.py run: an encoder over every row of a dataset."""
    encoder: CachedEncoder
    dataset: MulTaBenchDatasetID

    @property
    def path(self) -> str:
        return join(repo_name(self.encoder), embeddings_dir(self.dataset))

    def command(self, python: str, output_dir: str) -> str:
        return shlex.join([python, "embed.py", "--encoder", self.encoder, "--dataset_name", self.dataset.name,
                           "--output_dir", output_dir])


def list_jobs(encoders: Iterable[CachedEncoder], datasets: Iterable[MulTaBenchDatasetID]) -> List[EmbeddingJob]:
    return [EmbeddingJob(encoder, dataset) for encoder in encoders for dataset in datasets
            if embeds_dataset(encoder, dataset)]


def pending_jobs(jobs: List[EmbeddingJob], output_dir: str) -> List[EmbeddingJob]:
    return [job for job in jobs if not os.path.exists(join(output_dir, job.path, META_JSON))]
