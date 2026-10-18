"""Seconds per 1,000 rows each cached frozen encoder took on each dataset, from its meta.json on the Hub.

An estimate of what a run would pay to embed its rows live: the caches were computed on A100s, not the leaderboard's
H100, and the image time includes reading the files.
"""
import json
import os

import pandas as pd

from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.embeddings.hub import CACHED_ENCODERS, META_JSON, download_embeddings, embeds_dataset, encoder_spec

EMBEDDING_COST_CSV = os.path.join(os.path.dirname(__file__), "embedding_cost.csv")


def embedding_cost() -> pd.DataFrame:
    rows = []
    for encoder in [e for e in CACHED_ENCODERS if not encoder_spec(e).tune_encoder]:
        for dataset in MulTaBenchDatasetID:
            if not embeds_dataset(encoder, dataset):
                continue
            with open(os.path.join(download_embeddings(encoder, dataset), META_JSON)) as f:
                meta = json.load(f)
            rows.append({"dataset": dataset.name, "encoder": str(encoder),
                         "encode_s_per_1k": sum(meta["encode_seconds"].values()) / meta["n_rows"] * 1000,
                         "gpu": meta["gpu_type"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    embedding_cost().round(4).to_csv(EMBEDDING_COST_CSV, index=False)
    print(f"Wrote {EMBEDDING_COST_CSV}")
