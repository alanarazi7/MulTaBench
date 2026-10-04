import json

from multabench.benchmark.collect import RESULT_COLUMNS, collect_results
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
