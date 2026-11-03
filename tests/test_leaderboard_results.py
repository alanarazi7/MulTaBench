import glob
import os

import numpy as np
import pandas as pd
import pytest

from benchmark import BASELINES
from multabench.benchmark.collect import RESULT_COLUMNS
from multabench.benchmark.splits import SPLITS
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.leaderboard.elo import RESULTS_DIR
from multabench.leaderboard.export import ENCODING_DIRS, TEXT_ONLY_ENCODINGS
from multabench.result_keys import RUN_KEY, ResultKey, RunStatus
from multabench.utils.hardware import CPU_RUN_HARDWARE, GPU_RUN_HARDWARE

PATHS = [p for d in ENCODING_DIRS.values() for p in sorted(glob.glob(os.path.join(d, "*.csv")))]
MODELS = {model.MODEL_NAME: model for model in BASELINES}
TASK_METRICS = {"BIN": "roc_auc", "MUL": "log_loss", "REG": "root_mean_squared_error"}
# The runs a model can't do on the leaderboard's hardware, by model and dataset: out of GPU memory, or of RAM.
EXPECTED_FAILURES = {(f"TILE-Kumo-Tabular{size}", "MUL_TEXT_SPOTIFY_GENRES") for size in ["", "-M"]} | {
    (f"TILE-{model}", "MUL_TEXT_MELBOURNE_AIRBNB") for model in ["TabICLv2", "Xiaomi-TabLDM", "LimiX-2"]}


@pytest.fixture(scope="module", params=PATHS, ids=[os.path.relpath(p, os.path.dirname(RESULTS_DIR)) for p in PATHS])
def runs(request):
    """The model, its runs, and their encoding."""
    path = request.param
    encoding = next(e for e, d in ENCODING_DIRS.items() if os.path.dirname(path) == d)
    return os.path.basename(path)[:-len(".csv")], pd.read_csv(path), encoding


def test_a_file_holds_one_known_model_with_the_result_columns(runs):
    name, df, _ = runs
    assert list(df.columns) == RESULT_COLUMNS
    assert set(df[ResultKey.MODEL]) == {name} and name in MODELS


def test_every_dataset_has_every_fold_once(runs):
    """Every fold, but those of an expected failure, which a run killed for its memory leaves without a result."""
    name, df, encoding = runs
    assert set(df[ResultKey.SIZE]) == {"10k"}
    assert not df.duplicated(RUN_KEY).any()
    text_only = encoding in TEXT_ONLY_ENCODINGS
    datasets = [d for d in MulTaBenchDatasetID if not text_only or dataset_modality(d) == Modality.TEXT]
    expected = {(d.name, f) for d in datasets for f in range(SPLITS)}
    found = set(zip(df[ResultKey.DATASET], df[ResultKey.FOLD]))
    assert found <= expected
    assert {(name, dataset) for dataset, _ in expected - found} <= EXPECTED_FAILURES


def test_each_dataset_uses_the_encoders_of_its_columns(runs):
    """Small or large, frozen or -tar for the image column and a text-only dataset's text; image datasets' text stays
    frozen."""
    name, df, encoding = runs
    embeds_text = MODELS[name].USE_TEXT_EMBEDDINGS
    size = "large" if encoding.startswith("large") else "small"
    tar = "-tar" if encoding.endswith("tar") else ""
    for dataset, encoders in df.groupby(ResultKey.DATASET)[[ResultKey.TEXT_ENCODER, ResultKey.IMAGE_ENCODER]]:
        modality = dataset_modality(MulTaBenchDatasetID[dataset])
        text_tar = tar if modality == Modality.TEXT else ""
        text = f"e5-{size}{text_tar}" if modality.has_text and embeds_text else None
        image = f"dino-{size}{tar}" if modality.has_images else None
        assert set(encoders[ResultKey.TEXT_ENCODER].replace({np.nan: None})) == {text}, dataset
        assert set(encoders[ResultKey.IMAGE_ENCODER].replace({np.nan: None})) == {image}, dataset


def test_successful_runs_score_with_the_task_metric(runs):
    _, df, _ = runs
    ok = df[df[ResultKey.STATUS] == RunStatus.OK]
    assert (ok[ResultKey.METRIC] == ok[ResultKey.DATASET].str[:3].map(TASK_METRICS)).all()
    assert ok[[ResultKey.TEST_SCORE, ResultKey.TEST_ERROR]].notna().all().all() and ok[ResultKey.ERROR].isna().all()
    optimum = np.where(ok[ResultKey.METRIC] == "roc_auc", 1.0, 0.0)
    assert np.allclose(ok[ResultKey.TEST_SCORE], optimum - ok[ResultKey.TEST_ERROR])


def test_every_run_succeeded_but_the_expected_failures(runs):
    _, df, _ = runs
    failed = df[df[ResultKey.STATUS] != RunStatus.OK]
    assert set(zip(failed[ResultKey.MODEL], failed[ResultKey.DATASET])) <= EXPECTED_FAILURES


def test_runs_used_their_hardware(runs):
    name, df, _ = runs
    ok = df[df[ResultKey.STATUS] == RunStatus.OK]
    hardware = GPU_RUN_HARDWARE if MODELS[name].NEEDS_GPU else CPU_RUN_HARDWARE
    assert (ok[ResultKey.CPU_CORES] == hardware.cpus).all()
    assert ok[ResultKey.CPU_NAME].str.startswith(hardware.cpu_prefix).all()
    if hardware.gpu_prefix:
        assert ok[ResultKey.GPU_TYPE].str.startswith(hardware.gpu_prefix).all()
    else:
        assert ok[ResultKey.GPU_TYPE].isna().all()
