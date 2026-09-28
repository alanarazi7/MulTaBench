"""Task-type helpers for the MulTaBench datasets."""
TASK_REG = "reg"
TASK_CLS = "cls"


def task_type_from_name(dataset_id: str) -> str:
    prefix = dataset_id.split("_")[0]
    if prefix == "REG":
        return TASK_REG
    if prefix in ("BIN", "MUL"):
        return TASK_CLS
    raise ValueError(f"Cannot infer task type from dataset name: {dataset_id}")
