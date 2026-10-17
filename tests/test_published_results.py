from multabench import load_published_results
from multabench.benchmark.metrics import METRICS
from multabench.benchmark.collect import RESULT_COLUMNS
from multabench.datasets.info import dataset_info
from multabench.leaderboard.elo import load_runs
from multabench.result_keys import ResultKey, RunStatus


def test_published_results_are_the_model_csvs():
    df = load_published_results()
    assert list(df.columns) == RESULT_COLUMNS
    assert set(df[ResultKey.SIZE]) == {"10k"} and len(df) == len(load_runs())
    assert load_published_results("full").empty


def test_each_dataset_was_scored_with_its_task_metric():
    df = load_published_results()
    ok = df[df[ResultKey.STATUS] == RunStatus.OK]
    for dataset, metrics in ok.groupby(ResultKey.DATASET)[ResultKey.METRIC]:
        assert set(metrics) == {METRICS[dataset_info(dataset).task]}
