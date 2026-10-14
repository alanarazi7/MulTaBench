import glob
import os

import numpy as np
import pandas as pd
import pytest

from benchmark import BASELINES
from multabench.benchmark.collect import RESULT_COLUMNS
from multabench.benchmark.splits import SPLITS
from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
from multabench.result_keys import RUN_KEY, ResultKey, RunStatus
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
    assert set(df[ResultKey.MODEL]) == {name} and name in MODELS


def test_every_dataset_has_every_fold_once(runs):
    _, df = runs
    assert set(df[ResultKey.SIZE]) == {"10k"}
    assert not df.duplicated(RUN_KEY).any()
    expected = {(d.name, f) for d in MulTaBenchDatasetID for f in range(SPLITS)}
    assert set(zip(df[ResultKey.DATASET], df[ResultKey.FOLD])) == expected


def test_each_dataset_uses_the_frozen_encoders_of_its_columns(runs):
    name, df = runs
    embeds_text = MODELS[name].USE_TEXT_EMBEDDINGS
    for dataset, encoders in df.groupby(ResultKey.DATASET)[[ResultKey.TEXT_ENCODER, ResultKey.IMAGE_ENCODER]]:
        modality = dataset_modality(MulTaBenchDatasetID[dataset])
        text = "e5-small" if modality.has_text and embeds_text else None
        image = "dino-small" if modality.has_images else None
        assert set(encoders[ResultKey.TEXT_ENCODER].replace({np.nan: None})) == {text}, dataset
        assert set(encoders[ResultKey.IMAGE_ENCODER].replace({np.nan: None})) == {image}, dataset


def test_successful_runs_score_with_the_task_metric(runs):
    _, df = runs
    ok = df[df[ResultKey.STATUS] == RunStatus.OK]
    assert (ok[ResultKey.METRIC] == ok[ResultKey.DATASET].str[:3].map(TASK_METRICS)).all()
    assert ok[[ResultKey.TEST_SCORE, ResultKey.TEST_ERROR]].notna().all().all() and ok[ResultKey.ERROR].isna().all()
    optimum = np.where(ok[ResultKey.METRIC] == "roc_auc", 1.0, 0.0)
    assert np.allclose(ok[ResultKey.TEST_SCORE], optimum - ok[ResultKey.TEST_ERROR])


def test_the_only_failures_are_tabpfn_on_more_than_ten_classes(runs):
    name, df = runs
    failed = df[df[ResultKey.STATUS] != RunStatus.OK]
    assert (failed[ResultKey.STATUS] == RunStatus.ERROR).all()
    if name.startswith("TabPFN"):
        assert failed[ResultKey.ERROR].str.contains(TOO_MANY_CLASSES).all()
        assert set(failed[ResultKey.DATASET].str[:3]) <= {"MUL"}
    else:
        assert failed.empty


def test_runs_used_their_hardware(runs):
    name, df = runs
    ok = df[df[ResultKey.STATUS] == RunStatus.OK]
    hardware = GPU_RUN_HARDWARE if MODELS[name].NEEDS_GPU else CPU_RUN_HARDWARE
    assert (ok[ResultKey.CPU_CORES] == hardware.cpus).all()
    assert ok[ResultKey.CPU_NAME].str.startswith(hardware.cpu_prefix).all()
    if hardware.gpu_prefix:
        assert ok[ResultKey.GPU_TYPE].str.startswith(hardware.gpu_prefix).all()
    else:
        assert ok[ResultKey.GPU_TYPE].isna().all()
