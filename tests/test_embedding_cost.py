import pandas as pd

from multabench.leaderboard.embedding_cost import EMBEDDING_COST_CSV, with_tar_encode_times


def test_frozen_caches_take_the_tuned_caches_mean_encoding_time():
    caches = pd.DataFrame({"dataset": ["A", "A", "A", "B"], "encoder": ["e5-small", "e5-small-tar", "e5-small-tar",
                                                                        "e5-small"],
                           "fold": [None, 0, 1, None], "encode_s_per_1k": [9.0, 2.0, 4.0, 7.0]})
    assert with_tar_encode_times(caches)["encode_s_per_1k"].tolist() == [3.0, 2.0, 4.0, 7.0]


def test_large_frozen_caches_keep_their_own_encoding_time():
    caches = pd.DataFrame({"dataset": ["A", "A"], "encoder": ["e5-large", "e5-large-tar"], "fold": [None, 0],
                           "encode_s_per_1k": [9.0, 2.0]})
    assert with_tar_encode_times(caches)["encode_s_per_1k"].tolist() == [9.0, 2.0]


def test_every_frozen_image_cache_has_a_tuned_twin():
    caches = with_tar_encode_times(pd.read_csv(EMBEDDING_COST_CSV))
    frozen = caches[caches["encoder"] == "dino-small"].set_index("dataset")["encode_s_per_1k"]
    tuned = caches[caches["encoder"] == "dino-small-tar"].groupby("dataset")["encode_s_per_1k"].mean()
    assert frozen.sort_index().equals(tuned.loc[frozen.index].sort_index())
