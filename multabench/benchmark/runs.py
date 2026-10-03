import os


def result_path(output_dir: str, model: str, dataset: str, text_encoder: str, image_encoder: str, size: str, fold: int) -> str:
    return os.path.join(output_dir, f"{model}_{dataset}_{text_encoder}_{image_encoder}_{size}_{fold}.json")
