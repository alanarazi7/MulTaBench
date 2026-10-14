import glob
import os

import numpy as np
import pandas as pd
import pytest

from benchmark import BASELINES
from multabench.benchmark.collect import RESULT_COLUMNS
from multabench.benchmark.splits import SPLITS
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.result_keys import METRIC, STATUS, TEST_ERROR, TEST_SCORE, RunStatus
from multabench.utils.hardware import CPU_RUN_HARDWARE, GPU_RUN_HARDWARE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "multabench", "leaderboard", "models")
PATHS = sorted(glob.glob(os.path.join(MODELS_DIR, "*.csv")))
MODELS = {model.MODEL_NAME: model for model in BASELINES}
TASK_METRICS = {"BIN": "roc_auc", "MUL": "log_loss", "REG": "root_mean_squared_error"}
TOO_MANY_CLASSES = "exceeds the maximum number of classes"


@pytest.fixture(scope="module", params=PATHS, ids=[os.path.basename(p) for p in PATHS])
def runs(request):
    return os.path.basename(request.param)[:-len(".csv")], pd.read_csv(request.param)


def test_a_file_holds_one_known_model_with_the_result_columns(runs):
    name, df = runs
    assert list(df.columns) == RESULT_COLUMNS
    assert set(df["model"]) == {name} and name in MODELS


def test_every_dataset_has_every_fold_once(runs):
    _, df = runs
    assert set(df["size"]) == {"10k"}
    assert not df.duplicated(["dataset", "text_encoder", "image_encoder", "fold"]).any()
    assert set(zip(df["dataset"], df["fold"])) == {(d.name, f) for d in MulTaBenchDatasetID for f in range(SPLITS)}


def test_each_dataset_uses_the_frozen_encoders_of_its_columns(runs):
    name, df = runs
    embeds_text = MODELS[name].USE_TEXT_EMBEDDINGS
    for dataset, encoders in df.groupby("dataset")[["text_encoder", "image_encoder"]]:
        modality = dataset_modality(MulTaBenchDatasetID[dataset])
        text = "e5-small" if modality.has_text and embeds_text else None
        image = "dino-small" if modality.has_images else None
        assert set(encoders["text_encoder"].replace({np.nan: None})) == {text}, dataset
        assert set(encoders["image_encoder"].replace({np.nan: None})) == {image}, dataset


def test_successful_runs_score_with_the_task_metric(runs):
    _, df = runs
    ok = df[df[STATUS] == RunStatus.OK]
    assert (ok[METRIC] == ok["dataset"].str[:3].map(TASK_METRICS)).all()
    assert ok[[TEST_SCORE, TEST_ERROR]].notna().all().all() and ok["error"].isna().all()
    optimum = np.where(ok[METRIC] == "roc_auc", 1.0, 0.0)
    assert np.allclose(ok[TEST_SCORE], optimum - ok[TEST_ERROR])


def test_the_only_failures_are_tabpfn_on_more_than_ten_classes(runs):
    name, df = runs
    failed = df[df[STATUS] != RunStatus.OK]
    assert (failed[STATUS] == RunStatus.ERROR).all()
    if name.startswith("TabPFN"):
        assert failed["error"].str.contains(TOO_MANY_CLASSES).all()
        assert set(failed["dataset"].str[:3]) <= {"MUL"}
    else:
        assert failed.empty


def test_runs_used_their_hardware(runs):
    name, df = runs
    ok = df[df[STATUS] == RunStatus.OK]
    hardware = GPU_RUN_HARDWARE if MODELS[name].NEEDS_GPU else CPU_RUN_HARDWARE
    assert (ok["cpu_cores"] == hardware.cpus).all()
    assert ok["cpu_name"].str.startswith(hardware.cpu_prefix).all()
    if hardware.gpu_prefix:
        assert ok["gpu_type"].str.startswith(hardware.gpu_prefix).all()
    else:
        assert ok["gpu_type"].isna().all()
