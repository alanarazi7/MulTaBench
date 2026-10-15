import numpy as np
from pandas import DataFrame

from multabench.baselines.preprocessing.text_embeddings import TfidfColumnEncoder


def test_tfidf_embeddings_are_reproducible():
    rng = np.random.default_rng(0)
    words = [f"word{i}" for i in range(500)]
    x = DataFrame({"title": [" ".join(rng.choice(words, size=5)) for _ in range(300)]})

    def embed() -> np.ndarray:
        return TfidfColumnEncoder("title", x["title"]).embed(x)

    assert np.array_equal(embed(), embed())
