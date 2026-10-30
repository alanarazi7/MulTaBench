import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from multabench.baselines.preprocessing.text_embeddings import TileColumnEncoder, fit_text_encoders
from multabench.e5.constants import TextEncoder
from multabench.tile import tile_embed
from multabench.tile.tile_embed import TileRows, serialize_rows


def test_serialize_rows():
    x = DataFrame({"wine_name": ["Rioja ", None], "price": [12.5, np.nan], "notes": ["oak", " "]})
    texts, spans = serialize_rows(x)
    assert texts == ["The wine name is Rioja. The price is 12.5. The notes is oak.",
                     "The wine name is NaN. The price is NaN. The notes is NaN."]
    assert [texts[0][start:end] for start, end in spans[0]] == ["The wine name is Rioja.", "The price is 12.5.",
                                                                "The notes is oak."]


@pytest.fixture
def pooled(monkeypatch):
    """A fake LLM: each cell's embedding is its row's text length and the cell's start character."""
    calls = []

    def fake_pool(model, tokenizer, texts, spans):
        calls.append(texts)
        return np.stack([np.stack([np.full(len(span), len(text)), span[:, 0]], axis=-1)
                         for text, span in zip(texts, spans)])

    monkeypatch.setattr(tile_embed, "get_tile_llm", lambda device: (None, None))
    monkeypatch.setattr(tile_embed, "pool_cells", fake_pool)
    return calls


def test_text_columns_share_one_pass_per_rows(pooled):
    x = DataFrame({"title": ["a", "bb", "ccc"], "price": [1.0, 2.0, 3.0], "photo": ["1.jpg", "2.jpg", "3.jpg"],
                   "notes": ["x", "y", "z"]}, index=[10, 20, 30])
    rows = TileRows(image_column="photo", device="cpu")
    rows.add(x.iloc[:2])
    rows.add(x.iloc[2:])
    rows.text_columns = ["notes", "title"]
    title, notes = TileColumnEncoder("title", rows), TileColumnEncoder("notes", rows)
    texts, spans = serialize_rows(x.drop(columns="photo"))
    assert np.array_equal(notes.embed(x.iloc[:2])[:, 1], spans[:2, 2, 0])
    assert np.array_equal(title.embed(x.iloc[:2])[:, 1], spans[:2, 0, 0])
    assert len(pooled) == 1 and "jpg" not in pooled[0][0]
    assert np.array_equal(title.embed(x.iloc[2:])[:, 0], [len(texts[2])])
    notes.embed(x.iloc[2:])
    assert len(pooled) == 2


def test_tile_needs_the_rows():
    with pytest.raises(ValueError, match="whole rows"):
        fit_text_encoders(DataFrame({"title": ["a"]}), text_features={"title"}, device="cpu", encoder=TextEncoder.TILE)


def test_rows_need_a_unique_index():
    with pytest.raises(ValueError, match="unique"):
        TileRows(image_column=None, device="cpu").add(DataFrame({"title": ["a", "b"]}, index=pd.Index([0, 0])))
