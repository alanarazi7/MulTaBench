import argparse

from multabench.datasets.downloading import download_multimodal_dataset
from multabench.datasets.utils import dataset_from_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    dataset_id = dataset_from_name(name=args.dataset)
    download_multimodal_dataset(dataset_id=dataset_id, for_annotation=True)
