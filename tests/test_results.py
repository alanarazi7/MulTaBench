from multabench.leaderboard.results import all_results
from multabench.result_keys import ResultKey


def test_all_results_have_one_row_per_model_encoding_dataset_and_fold():
    df = all_results()
    assert not df.duplicated([ResultKey.MODEL, "encoding", ResultKey.DATASET, ResultKey.FOLD]).any()
    assert set(df["encoding"]) == {"frozen", "tar", "end-to-end"}
