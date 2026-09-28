from dataclasses import dataclass
from enum import Enum

from pandas import DataFrame, Series

from multabench.datasets.all_datasets import MulTaBenchDatasetID


class SupervisedTask(Enum):
    REGRESSION = "📈 regression"
    BINARY = "⚖️ binary"
    MULTICLASS = "🎨 multiclass"


@dataclass
class MultimodalDataset:
    x: DataFrame
    y: Series
    task_type: SupervisedTask
    dataset_id: MulTaBenchDatasetID
    image_folder: str | None = None

    @property
    def is_cls(self) -> bool:
        return self.task_type != SupervisedTask.REGRESSION
