import os
import shlex
from dataclasses import dataclass
from os.path import join
from typing import Iterable, List, Optional

from multabench.benchmark.splits import DEFAULT_SIZE, SPLITS, Size
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.embeddings.hub import META_JSON, CachedEncoder, embeddings_dir, embeds_dataset, is_tuned, repo_name


@dataclass(frozen=True)
class EmbeddingJob:
    """One multabench.embeddings.embed run: a frozen encoder over every row of a dataset, or a tuned one over a fold."""
    encoder: CachedEncoder
    dataset: MulTaBenchDatasetID
    size: Optional[Size] = None
    fold: Optional[int] = None

    @property
    def path(self) -> str:
        return join(repo_name(self.encoder), embeddings_dir(self.dataset, size=self.size, fold=self.fold))

    def command(self, python: str, output_dir: str) -> str:
        split = [] if self.fold is None else ["--size", self.size, "--fold", str(self.fold)]
        return shlex.join([python, "-m", "multabench.embeddings.embed", "--encoder", self.encoder, "--dataset_name",
                           self.dataset.name, *split, "--output_dir", output_dir])


def list_jobs(encoders: Iterable[CachedEncoder], datasets: Iterable[MulTaBenchDatasetID], size: Size = DEFAULT_SIZE,
              folds: Iterable[int] = range(SPLITS)) -> List[EmbeddingJob]:
    folds = list(folds)
    return [job for encoder in encoders for dataset in datasets if embeds_dataset(encoder, dataset)
            for job in ([EmbeddingJob(encoder, dataset, Size(size), fold) for fold in folds] if is_tuned(encoder)
                        else [EmbeddingJob(encoder, dataset)])]


def pending_jobs(jobs: List[EmbeddingJob], output_dir: str) -> List[EmbeddingJob]:
    return [job for job in jobs if not os.path.exists(join(output_dir, job.path, META_JSON))]
