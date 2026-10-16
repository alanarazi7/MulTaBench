"""The datasets' modality, task and tier, without downloading them."""
from dataclasses import dataclass
from enum import StrEnum
from typing import List, Optional

from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.datasets.hub import hf_repo_id
from multabench.datasets.objects import SupervisedTask

PREFIX2TASK = {"BIN": SupervisedTask.BINARY, "MUL": SupervisedTask.MULTICLASS, "REG": SupervisedTask.REGRESSION}


class Tier(StrEnum):
    CORE = "core"
    EXTENDED = "extended"


@dataclass(frozen=True)
class DatasetInfo:
    id: MulTaBenchDatasetID
    hf_repo: str
    modality: Modality
    task: SupervisedTask
    tier: Tier


def dataset_info(dataset: str | MulTaBenchDatasetID) -> DatasetInfo:
    if isinstance(dataset, str):
        dataset = MulTaBenchDatasetID[dataset]
    return DatasetInfo(id=dataset, hf_repo=hf_repo_id(dataset), modality=dataset_modality(dataset),
                       task=PREFIX2TASK[dataset.name.split("_")[0]], tier=Tier(dataset.value.split("-")[0]))


def list_datasets(modality: Optional[Modality | str] = None, task: Optional[SupervisedTask | str] = None,
                  tier: Optional[Tier | str] = None) -> List[DatasetInfo]:
    """The datasets matching every filter given; task is a SupervisedTask or its name, e.g. "binary"."""
    if isinstance(task, str):
        task = SupervisedTask[task.upper()]
    infos = [dataset_info(d) for d in MulTaBenchDatasetID]
    return [i for i in infos if (modality is None or i.modality == Modality(modality))
            and (task is None or i.task == task) and (tier is None or i.tier == Tier(tier))]
