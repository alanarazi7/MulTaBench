"""The Hugging Face commit each dataset and embeddings repo is read at, so a package version fixes the data.

After uploading to one of these repos, `python -m multabench.datasets.revisions` pins every repo to its current main.
"""
import json
from functools import lru_cache
from os.path import dirname, join

REVISIONS_JSON = join(dirname(__file__), "revisions.json")


@lru_cache(maxsize=1)
def _revisions() -> dict:
    with open(REVISIONS_JSON) as f:
        return json.load(f)


def revision(repo_id: str) -> str:
    return _revisions()[repo_id]


def pin_main():
    from huggingface_hub import HfApi

    from multabench.datasets.all_datasets import MulTaBenchDatasetID
    from multabench.datasets.hub import hf_repo_id
    from multabench.embeddings.hub import CACHED_ENCODERS, embeddings_repo_id

    api = HfApi()
    repos = [hf_repo_id(d) for d in MulTaBenchDatasetID] + [embeddings_repo_id(e) for e in CACHED_ENCODERS]
    pins = {repo: api.dataset_info(repo).sha for repo in repos}
    for repo, sha in sorted(pins.items()):
        if _revisions().get(repo) != sha:
            print(f"{repo}: {_revisions().get(repo)} -> {sha}")
    with open(REVISIONS_JSON, "w") as f:
        json.dump(pins, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    pin_main()
