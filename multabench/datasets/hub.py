"""MulTaBench datasets on the Hugging Face Hub: one dataset repo each, holding a typed data.parquet and metadata.json."""
from multabench.datasets.all_datasets import MulTaBenchDatasetID, is_benchmark_dataset, is_image_dataset

HF_ORG = "multabench"
DATA_PARQUET = "data.parquet"

HF_DATASETS = {
    MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES,
    MulTaBenchDatasetID.REG_IMAGE_KHAADI_CLOTHES,
}


def hf_repo_id(dataset_id: MulTaBenchDatasetID) -> str:
    tier = "core" if is_benchmark_dataset(dataset_id) else "extended"
    modality = "img" if is_image_dataset(dataset_id) else "text"
    task = "reg" if dataset_id.name.startswith("REG_") else "cls"
    name = dataset_id.value.removeprefix("multabench-full-").removeprefix("multabench-")
    return f"{HF_ORG}/{tier}-{modality}-{task}-{name}"
