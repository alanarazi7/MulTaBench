from os.path import join

from pandas import DataFrame, read_csv

from multabench.datasets.all_datasets import ALL_DATASETS, MultimodalDatasetID


def dataset_from_name(name: str) -> MultimodalDatasetID:
    if not isinstance(name, str):
        raise TypeError(f"Expected string argument, got {type(name)}")
    for dataset in ALL_DATASETS:
        if dataset.name == name:
            return dataset
    raise ValueError(f"Dataset ID: {name} not found in any known datasets.")


def load_csv(dir_path: str, filename: str, sep: str = ",") -> DataFrame:
    full_path = join(dir_path, filename)
    df = read_csv(full_path, sep=sep)
    return df
