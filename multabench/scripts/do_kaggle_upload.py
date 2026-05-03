"""
Upload a prepared dataset folder to Kaggle.

Requires:
    - kaggle CLI installed: pip install kaggle
    - credentials at: ~/.kaggle/kaggle.json

Usage:
    python do_kaggle_upload.py --dataset_dir kaggle_uploads/multabench-lol-skins
    python do_kaggle_upload.py --dataset_dir kaggle_uploads/multabench-lol-skins --message "v2: added missing images"
"""
import argparse

from multabench.benchmark.curation.upload import upload_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the prepared dataset folder (output of do_kaggle_prepare.py)")
    parser.add_argument("--message", type=str, default="Initial upload",
                        help="Version message when updating an existing dataset")
    args = parser.parse_args()

    upload_dataset(dataset_dir=args.dataset_dir, version_message=args.message)
