import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series
from tile import embeddings
from tile.embeddings import serialize

from multabench.datasets.objects import SupervisedTask
from multabench.embeddings import compute
from multabench.embeddings.hub import read_embeddings, tile_cells, write_embeddings
from multabench.interfaces.tile import rows as tile_rows
from multabench.interfaces.tile.models import TILE_MODELS, TileKumoTabular, TileLightGBM
from multabench.interfaces.tile.rows import MAX_CELL_CHARS, TileRows, tile_string_columns, tile_table


@pytest.fixture
def pooled(monkeypatch):
    """A fake LLM: each cell's embedding is its row's text length and the cell's start character."""
    calls = []

    def fake_pool(model, tokenizer, texts, spans):
        calls.append(texts)
        cells = np.stack([np.stack([np.full(len(span), len(text)), span[:, 0]], axis=-1)
                          for text, span in zip(texts, spans)])
        return cells.transpose(1, 0, 2)

    monkeypatch.setattr(tile_rows, "get_tile_llm", lambda device: (None, None))
    monkeypatch.setattr(embeddings, "pool_cells", fake_pool)
    return calls


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
    rows.string_columns, rows.cache = ["title"], read_embeddings(str(tmp_path))
    assert rows.cells(x.index[::-1])[0].tolist() == [[6, 7], [2, 3]]
    assert not pooled


def table(n=40):
    """A text column, a category column, a number and a date, with an image column TILE leaves out."""
    rng = np.random.default_rng(0)
    return DataFrame({"title": [f"title {i}" for i in range(n)],
                      "city": pd.Categorical(rng.choice(["Paris", "Lyon", "Nice"], n)),
                      "price": rng.normal(size=n), "listed": pd.date_range("2020-01-01", periods=n),
                      "photo": [f"{i}.jpg" for i in range(n)]}, index=np.arange(100, 100 + n))


def test_tile_reads_the_category_columns_as_strings():
    x = table()
    assert list(tile_table(x, image_column="photo").columns) == ["title", "city", "price", "listed"]
    assert tile_string_columns(x, image_column="photo") == ["title", "city"]
    assert serialize(tile_table(x, "photo"))[0] == serialize(x.drop(columns="photo"))[0]


class Recorder:
    """Stands in for the wrapped model: keeps the input TILE gives it."""

    def __init__(self, **kwargs):
        self.inputs = []

    def fit(self, x, y):
        self.inputs.append(x)

    def predict(self, x):
        self.inputs.append(x)
        return np.zeros(len(x))


def wrap(model_cls, monkeypatch, task=SupervisedTask.BINARY):
    monkeypatch.setattr(model_cls, "TABULAR_MODEL", Recorder)
    return model_cls(problem_type=task, device="cpu", image_column=None)


def test_the_wrapped_model_gets_every_string_column_embedded_and_the_bypass_features(pooled, monkeypatch):
    x, y = table().drop(columns="photo"), Series(np.tile(["no", "yes"], 20), index=table().index)
    tile = wrap(TileKumoTabular, monkeypatch)
    tile.fit(x.iloc[:30], y.iloc[:30])
    tile.predict(x.iloc[30:])
    train, test = tile.tabular_model.inputs
    assert [c for c in train.columns if "_txt_pca_" in c] == ["title_txt_pca_0", "title_txt_pca_1", "city_txt_pca_0",
                                                              "city_txt_pca_1"]
    assert "city_te_0" in train.columns and "title_te_0" not in train.columns
    categories = [c for c in train.columns if isinstance(train[c].dtype, pd.CategoricalDtype)]
    assert categories == ["title", "city"] and train.drop(columns=categories).dtypes.eq(float).all()
    assert list(test.columns) == list(train.columns) and list(test.index) == list(x.index[30:])
    assert len(pooled) == 2


def test_a_model_without_a_tile_learner_gets_no_native_categories(pooled, monkeypatch):
    x, y = table().drop(columns="photo"), Series(np.arange(40.0), index=table().index)
    tile = wrap(TileLightGBM, monkeypatch, task=SupervisedTask.REGRESSION)
    tile.fit(x, y)
    [train] = tile.tabular_model.inputs
    assert "city" not in train.columns and train.dtypes.eq(float).all()


def test_tile_fits_the_real_model_on_its_input(pooled):
    x, y = table().drop(columns="photo"), Series(np.tile(["no", "yes"], 20), index=table().index)
    tile = TileLightGBM(problem_type=SupervisedTask.BINARY, device="cpu")
    tile.fit(x.iloc[:30], y.iloc[:30])
    assert tile.tabular_model.text_features == set() and len(tile.predict(x.iloc[30:])) == 10
    assert set(tile.score_all_metrics(x.iloc[30:], y.iloc[30:])) >= {"test_score", "test_error"}


def test_tile_runs_on_text_datasets_only(pooled):
    tile = TileLightGBM(problem_type=SupervisedTask.BINARY, device="cpu", image_column="photo")
    with pytest.raises(ValueError, match="text datasets"):
        tile.fit(table(), Series(np.tile([0, 1], 20), index=table().index))


def test_each_tile_model_is_named_after_its_model_and_shares_its_venv():
    names = {m.TABULAR_MODEL.MODEL_NAME: (m.MODEL_NAME, m.SHORT_NAME) for m in TILE_MODELS}
    assert names["Kumo-Tabular"] == ("TILE-Kumo-Tabular", "tilekumo")
    assert all(m.ENV == m.TABULAR_MODEL.ENV and m.NEEDS_GPU == m.TABULAR_MODEL.NEEDS_GPU for m in TILE_MODELS)
    assert not any(m.USE_TEXT_EMBEDDINGS for m in TILE_MODELS)
