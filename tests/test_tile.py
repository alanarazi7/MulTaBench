import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Index
from tile import embeddings
from tile.embeddings import serialize

from multabench.embeddings import compute
from multabench.embeddings.hub import read_embeddings, tile_cells, write_embeddings
from multabench.encoders.constants import TextEncoder
from multabench.encoders.registry import make_encoder
from multabench.encoders.tile import tile_encoder
from multabench.encoders.tile.tile_encoder import MAX_CELL_CHARS, TileColumnEncoder, TileRows


@pytest.fixture
def pooled(monkeypatch):
    """A fake LLM: each cell's embedding is its row's text length and the cell's start character."""
    calls = []

    def fake_pool(model, tokenizer, texts, spans):
        calls.append(texts)
        cells = np.stack([np.stack([np.full(len(span), len(text)), span[:, 0]], axis=-1)
                          for text, span in zip(texts, spans)])
        return cells.transpose(1, 0, 2)

    monkeypatch.setattr(tile_encoder, "get_tile_llm", lambda device: (None, None))
    monkeypatch.setattr(embeddings, "pool_cells", fake_pool)
    return calls


def test_text_columns_share_one_pass_per_rows(pooled):
    x = DataFrame({"title": ["a", "bb", "ccc"], "price": [1.0, 2.0, 3.0], "photo": ["1.jpg", "2.jpg", "3.jpg"],
                   "notes": ["x", "y", "z"]}, index=[10, 20, 30])
    rows = TileRows(image_column="photo", device="cpu")
    rows.add(x.iloc[:2])
    rows.add(x.iloc[2:])
    rows.text_columns = ["notes", "title"]
    title, notes = TileColumnEncoder("title", rows), TileColumnEncoder("notes", rows)
    texts, spans = serialize(x.drop(columns="photo"))
    assert np.array_equal(notes.embed(x.iloc[:2])[:, 1], spans[:2, 2, 0])
    assert np.array_equal(title.embed(x.iloc[:2])[:, 1], spans[:2, 0, 0])
    assert len(pooled) == 1 and "jpg" not in pooled[0][0]
    assert np.array_equal(title.embed(x.iloc[2:])[:, 0], [len(texts[2])])
    notes.embed(x.iloc[2:])
    assert len(pooled) == 2


def test_new_rows_under_seen_labels_are_embedded_again(pooled):
    rows = TileRows(image_column=None, device="cpu")
    rows.text_columns = ["title"]
    rows.add(DataFrame({"title": ["a", "bb"]}))
    first = rows.embed(Index([0, 1]), "title")
    rows.add(DataFrame({"title": ["ccc", "dddd"]}))
    assert not np.array_equal(rows.embed(Index([0, 1]), "title"), first) and len(pooled) == 2


def test_tile_needs_the_rows():
    with pytest.raises(ValueError, match="whole rows"):
        make_encoder(TextEncoder.TILE, device="cpu").fit(DataFrame({"title": ["a"]}), columns=["title"])


def test_the_encoder_embeds_the_rows_it_saw(pooled):
    x = DataFrame({"title": ["a", "bb", "ccc"], "photo": ["1.jpg", "2.jpg", "3.jpg"]})
    tile = make_encoder(TextEncoder.TILE, device="cpu", image_column="photo")
    tile.see_rows(x)
    [title] = tile.column_encoders(x, columns=["title"], y=None, is_cls=True)
    assert np.array_equal(title.embed(x)[:, 0], [len(t) for t in serialize(x[["title"]])[0]])


def test_rows_need_a_unique_index():
    with pytest.raises(ValueError, match="unique"):
        TileRows(image_column=None, device="cpu").add(DataFrame({"title": ["a", "b"]}, index=pd.Index([0, 0])))


def test_long_values_are_cut():
    rows = TileRows(image_column=None, device="cpu")
    rows.add(DataFrame({"title": ["a" * (MAX_CELL_CHARS + 10), "b"], "price": [1.0, None]}))
    assert rows.texts[0] == serialize(DataFrame({"title": ["a" * MAX_CELL_CHARS], "price": [1.0]}))[0][0]
    assert rows.texts[1] == serialize(DataFrame({"title": ["b"], "price": [None]}))[0][0]


def test_cache_round_trip_keys_rows_and_keeps_float16(pooled, tmp_path):
    x = DataFrame({"title": ["a", "bb", "a"], "price": [1.0, 2.0, 1.0], "notes": ["x", "y", "x"]})
    inputs, cells, _ = compute._encode_rows(x, image_column=None, features=["notes", "title"], device="cpu")
    texts, spans = serialize(x)
    assert inputs == texts[:2] and cells.shape == (2, 2, 2)
    write_embeddings(str(tmp_path), inputs=inputs, embeddings=cells, meta={"features": ["notes", "title"]},
                     dtype=np.float16)
    cache = read_embeddings(str(tmp_path))
    assert cache.embeddings.dtype == np.float16
    title = tile_cells(cache, [texts[2], texts[1]], ["title"])[0]
    assert np.array_equal(title[:, 1], [spans[2, 0, 0], spans[1, 0, 0]])
    with pytest.raises(ValueError, match="no \\['price'\\]"):
        tile_cells(cache, texts, ["price"])


def test_rows_read_the_cache_without_the_llm(pooled, tmp_path):
    x = DataFrame({"title": ["a", "bb"], "notes": ["x", "y"]}, index=[5, 7])
    texts, _ = serialize(x)
    cells = np.arange(8, dtype=np.float16).reshape(2, 2, 2)
    write_embeddings(str(tmp_path), inputs=texts, embeddings=cells, meta={"features": ["notes", "title"]},
                     dtype=np.float16)
    rows = TileRows(image_column=None, device="cpu")
    rows.add(x)
    rows.text_columns, rows.cache = ["title"], read_embeddings(str(tmp_path))
    assert rows.embed(x.index[::-1], "title").tolist() == [[6, 7], [2, 3]]
    assert not pooled
