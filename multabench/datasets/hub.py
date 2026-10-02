"""Image datasets keep their images in images-*.zip shards, because the Hub limits the number of files per repo."""
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
