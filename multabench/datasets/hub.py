"""
MulTaBench datasets on the Hugging Face Hub: one dataset repo each, holding a typed data.parquet and metadata.json.

Image datasets also hold their images packed in images-*.zip shards (the Hub limits files per folder and per repo).
Their paths inside the shards are the paths in the image column, so extracting the shards next to data.parquet
restores the images/ folder.
"""
import os
import tempfile
import zipfile
from glob import glob
from os.path import join

from multabench.datasets.all_datasets import MulTaBenchDatasetID

HF_ORG = "multabench"
DATA_PARQUET = "data.parquet"
IMAGES_DIR = "images"
IMAGE_SHARDS = "images-*.zip"


def hf_repo_id(dataset_id: MulTaBenchDatasetID) -> str:
    return f"{HF_ORG}/{dataset_id.value}"


def extract_images(dir_path: str):
    images_dir = join(dir_path, IMAGES_DIR)
    shards = sorted(glob(join(dir_path, IMAGE_SHARDS)))
    if not shards or os.path.isdir(images_dir):
        return
    print(f"Extracting {len(shards)} image shards...")
    tmp_dir = tempfile.mkdtemp(dir=dir_path)
    for shard in shards:
        with zipfile.ZipFile(shard) as z:
            z.extractall(tmp_dir)
    os.rename(join(tmp_dir, IMAGES_DIR), images_dir)
    os.rmdir(tmp_dir)
