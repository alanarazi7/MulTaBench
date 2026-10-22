"""Writes the CSVs the leaderboard Space shows, computed from the per-model run CSVs."""
import argparse
import itertools
import os

import pandas as pd

from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.datasets.info import Tier, dataset_info
from multabench.leaderboard.elo import ANCHOR, RESULTS_DIR, TAR_RESULTS_DIR, error_table, leaderboard, load_runs
from multabench.leaderboard.embedding_cost import EMBEDDING_COST_CSV, with_tar_encode_times
from multabench.result_keys import ResultKey, RunStatus

SPACE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "space", "data")
TIERS = [t.value for t in Tier]
TASKS = ["classification", "regression"]
# The board knows two modalities: a dataset with images counts as image, whether or not it also has text.
MODALITIES = [Modality.TEXT.value, Modality.IMAGE.value]
ENCODING, ENTRY = "encoding", "entry"
# How a model reads text and images: frozen or fine-tuned (TAR) encoders, or end-to-end, reading raw text itself.
ENCODINGS = {"frozen": "Frozen", "tar": "TAR"}
END_TO_END = "end-to-end"


def dataset_tier(dataset: str) -> str:
    return dataset_info(dataset).tier.value


def dataset_task(dataset: str) -> str:
    return "regression" if dataset.startswith("REG_") else "classification"


def _subset_key(picked: list[str], values: list[str]) -> str:
    return "all" if len(picked) == len(values) else "+".join(picked)


def subsets(values: list[str]) -> list[list[str]]:
    return [list(c) for n in range(1, len(values) + 1) for c in itertools.combinations(values, n)]


def entries(frozen: pd.DataFrame, tar: pd.DataFrame) -> pd.DataFrame:
    """Every run with its encoding and leaderboard entry, e.g. "LimiX-2 (TAR)"; an end-to-end model is its own entry."""
    frozen, tar = frozen.assign(**{ENCODING: "frozen"}), tar.assign(**{ENCODING: "tar"})
    raw_text = frozen.groupby(ResultKey.MODEL)[ResultKey.TEXT_ENCODER].transform(lambda e: e.isna().all())
    frozen.loc[raw_text, ENCODING] = END_TO_END
    runs = pd.concat([frozen, tar], ignore_index=True)
    names = zip(runs[ResultKey.MODEL], runs[ENCODING])
    runs[ENTRY] = [m if e == END_TO_END else f"{m} ({ENCODINGS[e]})" for m, e in names]
    return runs


def board_modality(dataset: str) -> str:
    """The dataset's modality on the board: image if it has images, else text."""
    return (Modality.IMAGE if dataset_modality(MulTaBenchDatasetID[dataset]).has_images else Modality.TEXT).value


def _filters(runs: pd.DataFrame):
    """Each combination of tiers, tasks and modalities: its keys and the runs on its datasets."""
    datasets = runs[ResultKey.DATASET]
    tier, task = datasets.map(dataset_tier), datasets.map(dataset_task)
    modality = datasets.map(board_modality)
    for tiers, tasks, modalities in itertools.product(subsets(TIERS), subsets(TASKS), subsets(MODALITIES)):
        keep = tier.isin(tiers) & task.isin(tasks) & modality.isin(modalities)
        keys = {"tier": _subset_key(tiers, TIERS), "task": _subset_key(tasks, TASKS),
                "modality": _subset_key(modalities, MODALITIES)}
        yield keys, runs[keep]


def _by_entry(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.assign(**{ResultKey.MODEL: runs[ENTRY]})


def _with_model_and_encoding(df: pd.DataFrame, runs: pd.DataFrame, at: int) -> pd.DataFrame:
    """Renames the entry column and puts its model and encoding after it."""
    of = runs.drop_duplicates(ENTRY).set_index(ENTRY)
    df = df.rename(columns={ResultKey.MODEL: ENTRY})
    df.insert(at + 1, ResultKey.MODEL, df[ENTRY].map(of[ResultKey.MODEL]))
    df.insert(at + 2, ENCODING, df[ENTRY].map(of[ENCODING]))
    return df


def leaderboards(runs: pd.DataFrame, bootstraps: int = 1000) -> pd.DataFrame:
    """One Elo leaderboard per combination of tiers, tasks, modalities and encodings; end-to-end models are in each."""
    boards = []
    for keys, sub in _filters(runs):
        for encodings in subsets(list(ENCODINGS)):
            picked = sub[sub[ENCODING].isin(encodings + [END_TO_END])]
            anchor = f"{ANCHOR} ({ENCODINGS[encodings[0]]})"
            board = leaderboard(error_table(_by_entry(picked)), bootstraps=bootstraps, anchor=anchor)
            board = _with_model_and_encoding(board, picked, at=0)
            for i, (k, v) in enumerate({**keys, "encodings": _subset_key(encodings, list(ENCODINGS))}.items()):
                board.insert(i, k, v)
            boards.append(board)
    return pd.concat(boards, ignore_index=True)


def encoding_leaderboards(runs: pd.DataFrame, bootstraps: int = 1000) -> pd.DataFrame:
    """Elo of the encodings themselves: each battle is one model's two runs on a dataset fold, end-to-end models aside.

    Bootstrapping draws dataset-model pairs, so the intervals are narrower than if they drew datasets."""
    ok = runs[(runs[ResultKey.STATUS] == RunStatus.OK) & (runs[ENCODING] != END_TO_END)]
    boards = []
    for keys, sub in _filters(ok):
        table = sub.groupby([ResultKey.DATASET, ResultKey.MODEL, ResultKey.FOLD, ENCODING])[ResultKey.TEST_ERROR].mean()
        table = table.unstack(ENCODING).rename(columns=ENCODINGS).dropna()
        pairs = table.index.droplevel(ResultKey.FOLD).unique()
        level = table.index.get_level_values
        table.index = pd.MultiIndex.from_arrays([level(0) + "|" + level(1), level(2)])
        board = leaderboard(table, bootstraps=bootstraps, anchor=ENCODINGS["frozen"])
        board = board.rename(columns={ResultKey.MODEL: ENCODING}).drop(columns=["avg_rank", "normalized_error"])
        board["datasets"] = pairs.get_level_values(0).nunique()
        board.insert(board.columns.get_loc("datasets") + 1, "models", pairs.get_level_values(1).nunique())
        for i, (k, v) in enumerate(keys.items()):
            board.insert(i, k, v)
        boards.append(board)
    return pd.concat(boards, ignore_index=True)


def per_dataset(runs: pd.DataFrame) -> pd.DataFrame:
    ok = _by_entry(runs[runs[ResultKey.STATUS] == RunStatus.OK])
    df = ok.groupby([ResultKey.DATASET, ResultKey.MODEL]).agg(**{
        ResultKey.METRIC: (ResultKey.METRIC, "first"), ResultKey.TEST_ERROR: (ResultKey.TEST_ERROR, "mean"),
        "test_error_std": (ResultKey.TEST_ERROR, "std"), ResultKey.TEST_SCORE: (ResultKey.TEST_SCORE, "mean"),
        "folds": (ResultKey.FOLD, "nunique"), ResultKey.N_TRAIN: (ResultKey.N_TRAIN, "max")}).reset_index()
    df.insert(1, "modality", df[ResultKey.DATASET].map(board_modality))
    df.insert(1, "task", df[ResultKey.DATASET].map(dataset_task))
    df.insert(1, "tier", df[ResultKey.DATASET].map(dataset_tier))
    return _with_model_and_encoding(df, runs, at=4)


def cost(runs: pd.DataFrame, embedding_cost_csv: str = EMBEDDING_COST_CSV) -> pd.DataFrame:
    """Median train and predict time per 1K rows, which leave out reading the cached embeddings; the mean time a TAR
    run's encoders took to fine-tune per 1K training rows; and the mean time its encoders would take to embed 1K rows
    live, estimated from the caches' own encoding time, the tuned caches' for both encodings."""
    ok = _by_entry(runs[runs[ResultKey.STATUS] == RunStatus.OK])
    caches = with_tar_encode_times(pd.read_csv(embedding_cost_csv)).astype({"fold": "Int64"}).fillna({"fold": -1})
    caches = caches.set_index(["dataset", "encoder", "fold"])

    def per_run(col: str, field: str) -> pd.Series:
        fold = ok[ResultKey.FOLD].where(ok[col].astype(str).str.endswith("-tar"), -1)
        keys = zip(ok[ResultKey.DATASET], ok[col], fold)
        return pd.Series([caches[field].get(k, 0.0) for k in keys], index=ok.index)

    ok = ok.assign(text_encode=per_run(ResultKey.TEXT_ENCODER, "encode_s_per_1k"),
                   image_encode=per_run(ResultKey.IMAGE_ENCODER, "encode_s_per_1k"),
                   finetune=per_run(ResultKey.TEXT_ENCODER, "finetune_s_per_1k")
                   + per_run(ResultKey.IMAGE_ENCODER, "finetune_s_per_1k"))
    df = ok.groupby(ResultKey.MODEL).agg(
        train_s_per_1k=(ResultKey.TRAIN_TIME_PER_1K, "median"),
        inference_s_per_1k=(ResultKey.INFERENCE_TIME_PER_1K, "median"), finetune_s_per_1k=("finetune", "mean"),
        text_encode_s_per_1k=("text_encode", "mean"), image_encode_s_per_1k=("image_encode", "mean"),
        gpu=(ResultKey.GPU_TYPE, "first"), cpu=(ResultKey.CPU_NAME, "first")).reset_index()
    return _with_model_and_encoding(df.sort_values("train_s_per_1k", ignore_index=True), runs, at=0)


def export(out_dir: str, results_dir: str = RESULTS_DIR, tar_results_dir: str = TAR_RESULTS_DIR,
           bootstraps: int = 1000):
    os.makedirs(out_dir, exist_ok=True)
    runs = entries(load_runs(results_dir), load_runs(tar_results_dir))
    leaderboards(runs, bootstraps).round(4).to_csv(os.path.join(out_dir, "leaderboards.csv"), index=False)
    encoding_leaderboards(runs, bootstraps).round(4).to_csv(os.path.join(out_dir, "encodings.csv"), index=False)
    per_dataset(runs).round(6).to_csv(os.path.join(out_dir, "per_dataset.csv"), index=False)
    cost(runs).round(4).to_csv(os.path.join(out_dir, "cost.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the CSVs of the leaderboard Space.")
    parser.add_argument('--out_dir', type=str, default=SPACE_DATA_DIR)
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--tar_results_dir', type=str, default=TAR_RESULTS_DIR)
    args = parser.parse_args()
    export(args.out_dir, args.results_dir, args.tar_results_dir)
