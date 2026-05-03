"""
Prepare a curated dataset for Kaggle upload.

Produces:
    kaggle_uploads/<slug>/
        data.csv
        images/
        metadata.json
        dataset-metadata.json

Usage:
    python do_kaggle_prepare.py --dataset_name MUL_IMAGE_LEAGUE_OF_LEGENDS_SKIN_CATEGORY --slug multabench-lol-skins
"""
import argparse

from multabench.benchmark.curation.prepare import prepare_dataset

DEFAULT_OUTPUT_ROOT = "kaggle_uploads"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset ID name, e.g. MUL_IMAGE_LEAGUE_OF_LEGENDS_SKIN_CATEGORY")
    parser.add_argument("--slug", type=str, required=True,
                        help="Kaggle dataset slug, e.g. multabench-lol-skins (becomes Chico89/<slug>)")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT,
                        help=f"Root directory for output folders (default: {DEFAULT_OUTPUT_ROOT})")
    args = parser.parse_args()

    out = prepare_dataset(dataset_name=args.dataset_name, slug=args.slug, output_root=args.output_root)
    print(f"\nReady to upload. Run:\n  python do_kaggle_upload.py --dataset_dir {out}")
