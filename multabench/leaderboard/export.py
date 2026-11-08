"""Writes the CSVs the leaderboard Space shows, computed from the per-model run CSVs."""
import argparse
import itertools
import os
from collections import Counter

import pandas as pd

from multabench.benchmark.splits import SPLITS
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.datasets.info import Tier, dataset_info
from multabench.leaderboard.elo import (ANCHOR, CASE_RESULTS_DIR, LARGE_RESULTS_DIR, LARGE_TAR_RESULTS_DIR, RESULTS_DIR,
                                        TAR_RESULTS_DIR, TFIDF_RESULTS_DIR, TILE_RESULTS_DIR, error_table, leaderboard,
                                        load_runs)
from multabench.leaderboard.embedding_cost import EMBEDDING_COST_CSV, with_tar_encode_times
from multabench.result_keys import ResultKey, RunStatus

SPACE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "space", "data")
TIERS = [t.value for t in Tier]
TASKS = ["classification", "regression"]
# The board knows two modalities: a dataset with images counts as image, whether or not it also has text.
MODALITIES = [Modality.TEXT.value, Modality.IMAGE.value]
ENCODING, ENTRY = "encoding", "entry"
# How a model reads text and images: small or large encoders, frozen or fine-tuned (TAR), TF-IDF, TILE's or CASE's
# row-aware embeddings, or end-to-end on raw text.
ENCODINGS = {"frozen": "Frozen-S", "tar": "TAR-S", "large": "Frozen-L", "large-tar": "TAR-L", "tfidf": "TF-IDF",
             "tile": "TILE", "case": "CASE"}
# Each encoding's runs; the end-to-end models are among the frozen ones.
ENCODING_DIRS = {"frozen": RESULTS_DIR, "tar": TAR_RESULTS_DIR, "large": LARGE_RESULTS_DIR,
                 "large-tar": LARGE_TAR_RESULTS_DIR, "tfidf": TFIDF_RESULTS_DIR, "tile": TILE_RESULTS_DIR,
                 "case": CASE_RESULTS_DIR}
END_TO_END = "end-to-end"
TFIDF, TILE, CASE = "tfidf", "tile", "case"
# An interface's runs name the model it wraps after its prefix.
PREFIXES = {TILE: "TILE-", CASE: "CASE-"}
# TF-IDF, TILE and CASE run on the text datasets only, so they are rated only where the board shows no image datasets.
TEXT_ONLY_ENCODINGS = [TFIDF, TILE, CASE]
NOT_RUN = "Not run yet"
# Why an entry has no result on a dataset, by encoding, model and dataset; an entry with no result at all is not run yet.
MISSING_REASONS = {
    **{(TILE, f"Kumo-Tabular{size}", "MUL_TEXT_SPOTIFY_GENRES"): "Out of GPU memory in fit" for size in ["", "-M"]},
    **{(TILE, model, "MUL_TEXT_MELBOURNE_AIRBNB"): "Out of RAM in predict"
       for model in ["TabICLv2", "Xiaomi-TabLDM", "LimiX-2"]},
    (TFIDF, "Mitra-v2", "BIN_TEXT_OSHA_INJURY"): "Rate-limited by the Hugging Face Hub, to rerun",
    (CASE, "Kumo-Tabular", "MUL_TEXT_SPOTIFY_GENRES"): "Out of GPU memory in fit",
}


def dataset_tier(dataset: str) -> str:
    return dataset_info(dataset).tier.value


def dataset_task(dataset: str) -> str:
    return "regression" if dataset.startswith("REG_") else "classification"


def _subset_key(picked: list[str], values: list[str]) -> str:
    return "all" if len(picked) == len(values) else "+".join(picked)


def subsets(values: list[str]) -> list[list[str]]:
    return [list(c) for n in range(1, len(values) + 1) for c in itertools.combinations(values, n)]


def load_entries(dirs: dict[str, str] = ENCODING_DIRS) -> pd.DataFrame:
    return entries({encoding: load_runs(results_dir) for encoding, results_dir in dirs.items()})


def entries(runs_by_encoding: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every run with its encoding and leaderboard entry, e.g. "LimiX-2 (TAR-S)"; an end-to-end model is its own entry."""
    runs = pd.concat([r.assign(**{ENCODING: e}) for e, r in runs_by_encoding.items()], ignore_index=True)
    frozen = runs[runs[ENCODING] == "frozen"]
    raw_text = frozen.groupby(ResultKey.MODEL)[ResultKey.TEXT_ENCODER].transform(lambda e: e.isna().all())
    runs.loc[raw_text[raw_text].index, ENCODING] = END_TO_END
    for encoding, prefix in PREFIXES.items():
        wrapped = runs[ENCODING] == encoding
        runs.loc[wrapped, ResultKey.MODEL] = runs.loc[wrapped, ResultKey.MODEL].str.removeprefix(prefix)
    names = zip(runs[ResultKey.MODEL], runs[ENCODING])
    runs[ENTRY] = [m if e == END_TO_END else f"{m} ({ENCODINGS[e]})" for m, e in names]
    return runs


def encoding_datasets(encoding: str) -> list[str]:
    return [d.name for d in MulTaBenchDatasetID
            if encoding not in TEXT_ONLY_ENCODINGS or dataset_modality(d) == Modality.TEXT]


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
        if modalities != [Modality.TEXT.value]:
            keep &= ~runs[ENCODING].isin(TEXT_ONLY_ENCODINGS)
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
    """One Elo leaderboard per combination of tiers, tasks and modalities, with every entry; the Space shows a slice of
    it for the encodings picked, so an entry's Elo doesn't depend on which others are shown."""
    boards = []
    anchor = f"{ANCHOR} ({ENCODINGS['frozen']})"
    for keys, sub in _filters(runs):
        board = leaderboard(error_table(_by_entry(sub)), bootstraps=bootstraps, anchor=anchor)
        board = _with_model_and_encoding(board, sub, at=0)
        for i, (k, v) in enumerate(keys.items()):
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
        table = table.unstack(ENCODING)
        # A pair needs every encoding but the text-only ones, which lack the image datasets and some folds or models.
        table = table.dropna(subset=table.columns.difference(TEXT_ONLY_ENCODINGS)).rename(columns=ENCODINGS)
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
    """Per board, over its datasets: median train and predict time per 1K rows, which leave out reading the cached
    embeddings; the mean time a TAR run's encoders took to fine-tune per 1K training rows; and the mean time its
    encoders would take to embed 1K rows live, from the caches' own encoding time, the tuned caches' for both; TF-IDF,
    fitted in the run, from the run's own; CASE adds its per-fold cache's to the model's own text encoder's."""
    ok = _by_entry(runs[runs[ResultKey.STATUS] == RunStatus.OK])
    ok[ResultKey.TEXT_ENCODER] = ok[ResultKey.TEXT_ENCODER].mask(ok[ENCODING] == TILE, TILE)
    tfidf_encode = (ok[ResultKey.TRAIN_EMBEDDING_TIME_PER_1K] * ok[ResultKey.N_TRAIN]
                    + ok[ResultKey.INFERENCE_EMBEDDING_TIME_PER_1K] * ok[ResultKey.N_TEST]) / (
                        ok[ResultKey.N_TRAIN] + ok[ResultKey.N_TEST])
    caches = with_tar_encode_times(pd.read_csv(embedding_cost_csv)).astype({"fold": "Int64"}).fillna({"fold": -1})
    caches = caches.set_index(["dataset", "encoder", "fold"])

    def per_run(col: str, field: str) -> pd.Series:
        fold = ok[ResultKey.FOLD].where(ok[col].astype(str).str.endswith("-tar"), -1)
        keys = zip(ok[ResultKey.DATASET], ok[col], fold)
        return pd.Series([caches[field].get(k, 0.0) for k in keys], index=ok.index)

    text_encode = per_run(ResultKey.TEXT_ENCODER, "encode_s_per_1k").mask(ok[ENCODING] == TFIDF, tfidf_encode)
    case_keys = zip(ok[ResultKey.DATASET], [CASE] * len(ok), ok[ResultKey.FOLD])
    case_encode = pd.Series([caches["encode_s_per_1k"].get(k, 0.0) for k in case_keys], index=ok.index)
    text_encode += case_encode.where(ok[ENCODING] == CASE, 0.0)
    ok = ok.assign(text_encode=text_encode,
                   image_encode=per_run(ResultKey.IMAGE_ENCODER, "encode_s_per_1k"),
                   finetune=per_run(ResultKey.TEXT_ENCODER, "finetune_s_per_1k")
                   + per_run(ResultKey.IMAGE_ENCODER, "finetune_s_per_1k"))
    boards = []
    for keys, sub in _filters(ok):
        board = sub.groupby(ResultKey.MODEL).agg(
            train_s_per_1k=(ResultKey.TRAIN_TIME_PER_1K, "median"),
            inference_s_per_1k=(ResultKey.INFERENCE_TIME_PER_1K, "median"), finetune_s_per_1k=("finetune", "mean"),
            text_encode_s_per_1k=("text_encode", "mean"), image_encode_s_per_1k=("image_encode", "mean"),
            gpu=(ResultKey.GPU_TYPE, "first"), cpu=(ResultKey.CPU_NAME, "first")).reset_index()
        board = _with_model_and_encoding(board.sort_values("train_s_per_1k", ignore_index=True), runs, at=0)
        for i, (k, v) in enumerate(keys.items()):
            board.insert(i, k, v)
        boards.append(board)
    return pd.concat(boards, ignore_index=True)


def missing(runs: pd.DataFrame) -> pd.DataFrame:
    """The folds each entry of a model that embeds has no result on, per dataset with why, or in one row if none."""
    embedded = runs[runs[ENCODING] != END_TO_END]
    ok = embedded[embedded[ResultKey.STATUS] == RunStatus.OK]
    done = set(zip(ok[ResultKey.MODEL], ok[ENCODING], ok[ResultKey.DATASET], ok[ResultKey.FOLD]))
    rows = []
    for model, encoding in itertools.product(sorted(embedded[ResultKey.MODEL].unique()), ENCODINGS):
        datasets = encoding_datasets(encoding)
        lacking = Counter(d for d in datasets for f in range(SPLITS) if (model, encoding, d, f) not in done)
        if lacking.total() == len(datasets) * SPLITS:
            rows.append((model, encoding, None, lacking.total(), NOT_RUN))
        else:
            rows += [(model, encoding, d, n, MISSING_REASONS[encoding, model, d]) for d, n in sorted(lacking.items())]
    df = pd.DataFrame(rows, columns=[ResultKey.MODEL, ENCODING, ResultKey.DATASET, "folds", "reason"])
    df.insert(0, ENTRY, [f"{m} ({ENCODINGS[e]})" for m, e in zip(df[ResultKey.MODEL], df[ENCODING])])
    for name, of in [("tier", dataset_tier), ("task", dataset_task), ("modality", board_modality)]:
        df.insert(df.columns.get_loc("folds"), name, df[ResultKey.DATASET].map(of, na_action="ignore"))
    return df


def export(out_dir: str, dirs: dict[str, str] = ENCODING_DIRS, bootstraps: int = 1000):
    os.makedirs(out_dir, exist_ok=True)
    runs = load_entries(dirs)
    leaderboards(runs, bootstraps).round(4).to_csv(os.path.join(out_dir, "leaderboards.csv"), index=False)
    encoding_leaderboards(runs, bootstraps).round(4).to_csv(os.path.join(out_dir, "encodings.csv"), index=False)
    per_dataset(runs).round(6).to_csv(os.path.join(out_dir, "per_dataset.csv"), index=False)
    cost(runs).round(4).to_csv(os.path.join(out_dir, "cost.csv"), index=False)
    missing(runs).to_csv(os.path.join(out_dir, "missing.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the CSVs of the leaderboard Space.")
    parser.add_argument('--out_dir', type=str, default=SPACE_DATA_DIR)
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--tar_results_dir', type=str, default=TAR_RESULTS_DIR)
    parser.add_argument('--large_results_dir', type=str, default=LARGE_RESULTS_DIR)
    parser.add_argument('--large_tar_results_dir', type=str, default=LARGE_TAR_RESULTS_DIR)
    parser.add_argument('--tfidf_results_dir', type=str, default=TFIDF_RESULTS_DIR)
    parser.add_argument('--tile_results_dir', type=str, default=TILE_RESULTS_DIR)
    parser.add_argument('--case_results_dir', type=str, default=CASE_RESULTS_DIR)
    args = parser.parse_args()
    export(args.out_dir, {"frozen": args.results_dir, "tar": args.tar_results_dir, "large": args.large_results_dir,
                          "large-tar": args.large_tar_results_dir, "tfidf": args.tfidf_results_dir,
                          "tile": args.tile_results_dir, "case": args.case_results_dir})
