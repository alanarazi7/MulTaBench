import glob
import os

import numpy as np
import pandas as pd
import pytest

from benchmark import BASELINES
from multabench.benchmark.collect import RESULT_COLUMNS
from multabench.benchmark.splits import SPLITS
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.leaderboard.elo import RESULTS_DIR, TAR_RESULTS_DIR
from multabench.result_keys import RUN_KEY, ResultKey, RunStatus
from multabench.utils.hardware import CPU_RUN_HARDWARE, GPU_RUN_HARDWARE

PATHS = [p for d in (RESULTS_DIR, TAR_RESULTS_DIR) for p in sorted(glob.glob(os.path.join(d, "*.csv")))]
MODELS = {model.MODEL_NAME: model for model in BASELINES}
TASK_METRICS = {"BIN": "roc_auc", "MUL": "log_loss", "REG": "root_mean_squared_error"}
TOO_MANY_CLASSES = ["MUL_IMAGE_HEARTHSTONE_CLASS", "MUL_TEXT_SPOTIFY_GENRES", "MUL_TEXT_WINE_REVIEW"]
# The only failures: the models that take at most 10 classes, and Mitra-v2 out of GPU memory on 680 features.
EXPECTED_FAILURES = {
    "TabPFN-v2": dict.fromkeys(TOO_MANY_CLASSES, "exceeds the maximum number of classes"),
    "TabPFN-v2p5": dict.fromkeys(TOO_MANY_CLASSES, "exceeds the maximum number of classes"),
    "Mitra-v2": {**dict.fromkeys(TOO_MANY_CLASSES, "supports at most 10 classes"),
                 "MUL_TEXT_MELBOURNE_AIRBNB": "No models were trained successfully"},
}


@pytest.fixture(scope="module", params=PATHS, ids=[os.path.relpath(p, os.path.dirname(RESULTS_DIR)) for p in PATHS])
def runs(request):
    """The model, its runs, and whether its encoders were fine-tuned (-tar)."""
    path = request.param
    return os.path.basename(path)[:-len(".csv")], pd.read_csv(path), os.path.dirname(path) == TAR_RESULTS_DIR


def test_a_file_holds_one_known_model_with_the_result_columns(runs):
    name, df, _ = runs
    assert list(df.columns) == RESULT_COLUMNS
    assert set(df[ResultKey.MODEL]) == {name} and name in MODELS


def test_every_dataset_has_every_fold_once(runs):
    _, df, _ = runs
    assert set(df[ResultKey.SIZE]) == {"10k"}
    assert not df.duplicated(RUN_KEY).any()
    expected = {(d.name, f) for d in MulTaBenchDatasetID for f in range(SPLITS)}
    assert set(zip(df[ResultKey.DATASET], df[ResultKey.FOLD])) == expected


def test_each_dataset_uses_the_encoders_of_its_columns(runs):
    """Frozen, or -tar for the image column and a text-only dataset's text; image datasets' text stays frozen."""
    name, df, tuned = runs
    embeds_text = MODELS[name].USE_TEXT_EMBEDDINGS
    for dataset, encoders in df.groupby(ResultKey.DATASET)[[ResultKey.TEXT_ENCODER, ResultKey.IMAGE_ENCODER]]:
        modality = dataset_modality(MulTaBenchDatasetID[dataset])
        text_tar = "-tar" if tuned and modality == Modality.TEXT else ""
        text = f"e5-small{text_tar}" if modality.has_text and embeds_text else None
        image = f"dino-small{'-tar' if tuned else ''}" if modality.has_images else None
        assert set(encoders[ResultKey.TEXT_ENCODER].replace({np.nan: None})) == {text}, dataset
        assert set(encoders[ResultKey.IMAGE_ENCODER].replace({np.nan: None})) == {image}, dataset


def test_successful_runs_score_with_the_task_metric(runs):
    _, df, _ = runs
    ok = df[df[ResultKey.STATUS] == RunStatus.OK]
    assert (ok[ResultKey.METRIC] == ok[ResultKey.DATASET].str[:3].map(TASK_METRICS)).all()
    assert ok[[ResultKey.TEST_SCORE, ResultKey.TEST_ERROR]].notna().all().all() and ok[ResultKey.ERROR].isna().all()
    optimum = np.where(ok[ResultKey.METRIC] == "roc_auc", 1.0, 0.0)
    assert np.allclose(ok[ResultKey.TEST_SCORE], optimum - ok[ResultKey.TEST_ERROR])


def test_the_only_failures_are_the_expected_ones(runs):
    name, df, _ = runs
    failed = df[df[ResultKey.STATUS] != RunStatus.OK]
    assert (failed[ResultKey.STATUS] == RunStatus.ERROR).all()
    expected = EXPECTED_FAILURES.get(name, {})
    assert sorted(failed[ResultKey.DATASET].unique()) == sorted(expected)
    for dataset, error in expected.items():
        errors = failed[failed[ResultKey.DATASET] == dataset][ResultKey.ERROR].astype(str)
        assert len(errors) == SPLITS and errors.str.contains(error).all()


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
