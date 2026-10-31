"""Seconds per 1,000 rows each cached encoder took on each dataset, from its meta.json on the Hub.

A frozen encoder has one cache per dataset; a tuned (-tar) one has one per fold, with the time fine-tuning took on the
fold's training rows. The encoding time estimates what a run would pay to embed its rows live: the small frozen caches
were computed on A100s, the tuned and large ones on H100s, and the image time includes reading the files. The board
charges small frozen runs the tuned caches' time, see `with_tar_encode_times`.
"""
import json
import os

import pandas as pd

from multabench.benchmark.splits import DEFAULT_SIZE, SPLITS
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.encoders.constants import ImageEncoder, TextEncoder
from multabench.embeddings.hub import CACHED_ENCODERS, META_JSON, download_embeddings, embeds_dataset, is_tuned

EMBEDDING_COST_CSV = os.path.join(os.path.dirname(__file__), "embedding_cost.csv")
TAR_SUFFIX = "-tar"
SMALL_TAR_ENCODERS = (str(TextEncoder.E5_SMALL_TAR), str(ImageEncoder.DINO_SMALL_TAR))


def embedding_cost() -> pd.DataFrame:
    rows = []
    for encoder in CACHED_ENCODERS:
        for dataset in MulTaBenchDatasetID:
            if not embeds_dataset(encoder, dataset):
                continue
            for fold in range(SPLITS) if is_tuned(encoder) else [None]:
                split = dict(size=DEFAULT_SIZE, fold=fold) if fold is not None else {}
                path = download_embeddings(encoder, dataset, **split, files=[META_JSON])
                with open(os.path.join(path, META_JSON)) as f:
                    meta = json.load(f)
                finetune = float(meta.get("finetune_seconds", 0.0)) / int(meta.get("n_train", 1)) * 1000
                rows.append({"dataset": dataset.name, "encoder": str(encoder), "fold": fold,
                             "encode_s_per_1k": sum(meta["encode_seconds"].values()) / int(meta["n_rows"]) * 1000,
                             "finetune_s_per_1k": finetune if fold is not None else 0.0, "gpu": meta["gpu_type"]})
    return pd.DataFrame(rows).astype({"fold": "Int64"})


def with_tar_encode_times(caches: pd.DataFrame) -> pd.DataFrame:
    """A hack: each small frozen cache takes its tuned twin's mean encoding time over the folds, on the same dataset.

    Frozen and tuned encoders embed with the same model, but the small frozen caches were timed on A100s and the tuned
    ones on H100s, so frozen looked about twice as slow. The large caches were all timed on H100s and keep their own
    time, as does a small frozen encoder with no tuned twin on a dataset.
    Once the small frozen encoders are re-timed on an H100, this can go.
    """
    tuned = caches[caches["encoder"].isin(SMALL_TAR_ENCODERS)]
    tar_time = tuned.groupby([tuned["encoder"].str.removesuffix(TAR_SUFFIX), "dataset"])["encode_s_per_1k"].mean()
    twin = tar_time.reindex(pd.MultiIndex.from_frame(caches[["encoder", "dataset"]])).set_axis(caches.index)
    return caches.assign(encode_s_per_1k=twin.fillna(caches["encode_s_per_1k"]))


if __name__ == "__main__":
    embedding_cost().round(4).to_csv(EMBEDDING_COST_CSV, index=False)
    print(f"Wrote {EMBEDDING_COST_CSV}")
