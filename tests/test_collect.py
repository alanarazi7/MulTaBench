import json

import pandas as pd

from multabench.benchmark.collect import RESULT_COLUMNS, collect_results, write_model_csvs
from multabench.result_keys import ResultKey, RunStatus


def _write(path, row):
    with open(path, "w") as f:
        json.dump(row, f)


def test_collect_keeps_failed_runs_and_the_result_columns(tmp_path):
    run = {ResultKey.MODEL: "LightGBM", ResultKey.FOLD: 0}
    _write(tmp_path / "ok.json", {**run, ResultKey.DATASET: "A", ResultKey.STATUS: RunStatus.OK,
                                  ResultKey.METRIC: "roc_auc", ResultKey.TEST_SCORE: 0.9, ResultKey.RUNTIME: 12.0})
    _write(tmp_path / "failed.json", {**run, ResultKey.DATASET: "B", ResultKey.STATUS: RunStatus.ERROR,
                                      ResultKey.ERROR: "ValueError: bad"})
    df = collect_results(str(tmp_path))
    assert list(df.columns) == RESULT_COLUMNS
    assert sorted(df[ResultKey.STATUS]) == [RunStatus.ERROR, RunStatus.OK]
    assert df.loc[df[ResultKey.DATASET] == "A", ResultKey.TEST_SCORE].item() == 0.9
    assert df.loc[df[ResultKey.DATASET] == "B", ResultKey.TEST_SCORE].isna().item()


def test_each_model_gets_one_csv_and_a_rerun_replaces_its_row(tmp_path):
    def runs(score, fold):
        return pd.DataFrame([{ResultKey.MODEL: m, ResultKey.DATASET: "A", ResultKey.TEXT_ENCODER: "e5-small",
                              ResultKey.IMAGE_ENCODER: None, ResultKey.SIZE: "10k", ResultKey.FOLD: fold,
                              ResultKey.TEST_SCORE: score} for m in ("LightGBM", "CatBoost")])
    write_model_csvs(pd.concat([runs(0.5, 0), runs(0.6, 1)]), str(tmp_path))
    write_model_csvs(runs(0.9, 0), str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["CatBoost.csv", "LightGBM.csv"]
    df = pd.read_csv(tmp_path / "LightGBM.csv")
    assert df.sort_values(ResultKey.FOLD)[ResultKey.TEST_SCORE].tolist() == [0.9, 0.6]
