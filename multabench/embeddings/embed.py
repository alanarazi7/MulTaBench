import argparse
import os
import sys

from multabench.benchmark.load import load_multabench_dataset
from multabench.benchmark.splits import DEFAULT_SIZE, SPLITS, Size
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.embeddings.compute import compute_embeddings
from multabench.embeddings.hub import CACHED_ENCODERS, META_JSON, embeds_dataset, is_tuned, parse_encoder
from multabench.embeddings.jobs import EmbeddingJob
from multabench.utils.devices import get_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed every row of one dataset with a frozen encoder, or a fold's "
                                                 "rows with a tuned (-tar) one, fine-tuned on its training rows.")
    parser.add_argument('--encoder', type=parse_encoder, required=True, choices=CACHED_ENCODERS)
    parser.add_argument('--dataset_name', type=str, required=True, choices=[d.name for d in MulTaBenchDatasetID])
    parser.add_argument('--fold', type=int, default=None, choices=range(SPLITS), help="for a tuned encoder")
    parser.add_argument('--size', type=Size, default=DEFAULT_SIZE, choices=list(Size), help="for a tuned encoder")
    parser.add_argument('--output_dir', type=str, default='embeddings')
    parser.add_argument('--device', type=str, default=None, help="e.g. cuda:1 or cpu; default: cuda, then mps, then cpu")
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()

    dataset = MulTaBenchDatasetID[args.dataset_name]
    if not embeds_dataset(args.encoder, dataset):
        parser.error(f"The benchmark doesn't use {args.encoder} on {dataset.name}")
    if is_tuned(args.encoder) != (args.fold is not None):
        parser.error(f"--fold is {'required' if is_tuned(args.encoder) else 'only'} for a tuned encoder")
    job = EmbeddingJob(args.encoder, dataset, *((args.size, args.fold) if is_tuned(args.encoder) else ()))
    path = os.path.join(args.output_dir, job.path)
    if os.path.exists(os.path.join(path, META_JSON)) and not args.overwrite:
        print(f"Skipping: {path} exists (--overwrite to recompute)")
        sys.exit(0)
    compute_embeddings(load_multabench_dataset(dataset), encoder=args.encoder, path=path,
                       device=get_device(device=args.device), size=job.size, fold=job.fold)
    print(f"Wrote {path}")
