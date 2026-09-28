from multabench.datasets.all_datasets import ALL_DATASETS, MultimodalDatasetID


def dataset_from_name(name: str) -> MultimodalDatasetID:
    if not isinstance(name, str):
        raise TypeError(f"Expected string argument, got {type(name)}")
    for dataset in ALL_DATASETS:
        if dataset.name == name:
            return dataset
    raise ValueError(f"Dataset ID: {name} not found in any known datasets.")
