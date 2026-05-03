"""
Upload a prepared dataset folder to Kaggle.

Expects the folder to already contain:
    data.csv, images/, metadata.json, dataset-metadata.json

Uses the `kaggle` CLI, which reads credentials from ~/.kaggle/kaggle.json.

Usage:
    python do_kaggle_upload.py --dataset_dir kaggle_uploads/multabench-league-of-legends-skin-category
"""
import subprocess
from os.path import join, exists

from multabench.benchmark.utils.constants import KAGGLE_METADATA_JSON


def upload_dataset(dataset_dir: str, version_message: str = "Initial upload") -> None:
    assert exists(dataset_dir), f"Dataset dir not found: {dataset_dir}"
    assert exists(join(dataset_dir, KAGGLE_METADATA_JSON)), \
        f"Missing {KAGGLE_METADATA_JSON} in {dataset_dir} — run do_kaggle_prepare.py first"

    # Try creating a new dataset first; if it already exists, create a new version instead
    print(f"Uploading {dataset_dir} to Kaggle (public)...")
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", dataset_dir, "--dir-mode", "zip", "-u"],
        capture_output=True, text=True,
    )
    combined = (result.stdout + result.stderr).lower()
    created_ok = result.returncode == 0 and "being created" in combined and "already in use" not in combined
    if created_ok:
        print(f"Dataset created successfully.\n{result.stdout}")
        return

    # Dataset already exists — push a new version
    if "already exists" in combined or "already in use" in combined:
        print("Dataset already exists, pushing new version...")
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", dataset_dir, "-m", version_message, "--dir-mode", "zip"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"New version uploaded successfully.\n{result.stdout}")
            return

    raise RuntimeError(f"Kaggle upload failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
