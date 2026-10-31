import numpy as np
from pandas import DataFrame

from multabench.encoders.constants import TextEncoder
from multabench.encoders.registry import make_encoder


def test_tfidf_embeddings_are_reproducible():
    rng = np.random.default_rng(0)
    words = [f"word{i}" for i in range(500)]
    x = DataFrame({"title": [" ".join(rng.choice(words, size=5)) for _ in range(300)]})

    def embed() -> np.ndarray:
        encoders, _ = make_encoder(TextEncoder.TFIDF, device="cpu").fit(x, columns=["title"])
        return encoders["title"].encode(x["title"])

    assert np.array_equal(embed(), embed())
