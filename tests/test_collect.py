import json

import pandas as pd

from multabench.benchmark.collect import RESULT_COLUMNS, collect_results, write_model_csvs
from multabench.result_keys import METRIC, STATUS, TEST_SCORE, RunStatus


def _write(path, row):
    with open(path, "w") as f:
        json.dump(row, f)


def test_collect_keeps_failed_runs_and_the_result_columns(tmp_path):
    _write(tmp_path / "ok.json", {"model": "LightGBM", "dataset": "A", "fold": 0, STATUS: RunStatus.OK,
                                   METRIC: "roc_auc", TEST_SCORE: 0.9, "runtime": 12.0})
    _write(tmp_path / "failed.json", {"model": "LightGBM", "dataset": "B", "fold": 0, STATUS: RunStatus.ERROR,
                                       "error": "ValueError: bad"})
    df = collect_results(str(tmp_path))
    assert list(df.columns) == RESULT_COLUMNS
    assert sorted(df[STATUS]) == [RunStatus.ERROR, RunStatus.OK]
    assert df.loc[df["dataset"] == "A", TEST_SCORE].item() == 0.9
    assert df.loc[df["dataset"] == "B", TEST_SCORE].isna().item()


def test_each_model_gets_one_csv_and_a_rerun_replaces_its_row(tmp_path):
    def runs(score, fold):
        return pd.DataFrame([{"model": m, "dataset": "A", "text_encoder": "e5-small", "image_encoder": None,
                              "size": "10k", "fold": fold, TEST_SCORE: score} for m in ("LightGBM", "CatBoost")])
    write_model_csvs(pd.concat([runs(0.5, 0), runs(0.6, 1)]), str(tmp_path))
    write_model_csvs(runs(0.9, 0), str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["CatBoost.csv", "LightGBM.csv"]
    df = pd.read_csv(tmp_path / "LightGBM.csv")
    assert df.sort_values("fold")[TEST_SCORE].tolist() == [0.9, 0.6]
