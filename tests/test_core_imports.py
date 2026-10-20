import subprocess
import sys

# Not psutil, which joblib imports when it's installed, nor the bare `autogluon` namespace its .pth files register.
BENCH_ONLY = ["torch", "torchvision", "transformers", "peft", "PIL", "skrub", "autogluon.core", "dotenv"]


def test_the_core_imports_no_bench_package():
    code = ("import sys, multabench; from multabench import (DATA_PARQUET, EMBEDDINGS_NPZ, FOLDS, META_JSON, "
            "METADATA_JSON, REPEATS, SPLITS, CachedEncoder, DatasetInfo, Modality, MulTaBenchDatasetID, Score, Size, "
            "SupervisedTask, Tier, data_dir_files, embeddings_repo_id, embeds_dataset, hf_repo_id, list_datasets, "
            "load_embeddings, load_published_results, load_split, parse_encoder, repo_name, revision, score); "
            f"print([m for m in {BENCH_ONLY} if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
