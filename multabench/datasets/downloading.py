from multabench.benchmark.load import load_multabench_dataset
from multabench.datasets.all_datasets import MultimodalDatasetID
from multabench.datasets.objects import MultimodalDataset


def download_dataset(dataset_id: MultimodalDatasetID) -> MultimodalDataset:
    return load_multabench_dataset(dataset_id)
