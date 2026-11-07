import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from multabench.benchmark.splits import Size
from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.objects import SupervisedTask
from multabench.embeddings import compute
from multabench.encoders.constants import TextEncoder
from multabench.embeddings.hub import (CACHED_ENCODERS, DEFAULT_ENCODERS, embeddings_dir, embeds_dataset, is_tuned,
                                       read_embeddings, write_embeddings)
from multabench.interfaces.case import case_interface
from multabench.interfaces.case.models import CASE_MODELS, CaseKumoTabular, CaseRandomForest
from multabench.interfaces.case.rows import case_row_keys, case_rows
from multabench.interfaces.constants import Interface

TEXT = MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING


def table(n=40):
    rng = np.random.default_rng(0)
    return DataFrame({"title": [f"title {i}" for i in range(n)],
                      "city": pd.Categorical(rng.choice(["Paris", "Lyon", "Nice"], n)),
                      "price": rng.normal(size=n)}, index=np.arange(100, 100 + n))


def test_keys_clean_cells_as_case_does():
    x = DataFrame({"title": ["a\tb\nc", "x    y", None], "n": [1, 2, 3]})
    assert case_row_keys(x) == ["abc\t1", "xy\t2", "\t3"]


def test_keys_match_cases_own_cleaning():
    core = pytest.importorskip("case.core")
    from case.config import SerializationConfig

    class Echo:
        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [texts] if isinstance(texts, str) else [[t] for t in texts]}

    x = DataFrame({"title": ["a\tb\nc", "x    y", None], "city": pd.Categorical(["P", None, "L"]), "n": [1.5, 2, None]})
    for (_, row), key in zip(x.iterrows(), case_row_keys(x)):
        cells = core._clean_and_tokenize_series(row, Echo(), SerializationConfig(cell_separator="\t"))
        assert "".join(cells[:-1]) == key


def test_case_is_cached_per_fold_on_text_datasets_only():
    assert Interface.CASE in CACHED_ENCODERS and Interface.CASE not in DEFAULT_ENCODERS and is_tuned(Interface.CASE)
    assert embeds_dataset(Interface.CASE, TEXT)
    assert not embeds_dataset(Interface.CASE, MulTaBenchDatasetID.MUL_IMAGE_PETFINDER)
    assert embeddings_dir(TEXT, size=Size.TEN_K, fold=2) == f"{TEXT.value}/10k/fold_2"


def fake_case(monkeypatch):
    """A fake CASE: each row's embedding is its price and its title's length, plus noise dimensions."""
    def embed(case, x):
        rng = np.random.default_rng(len(x))
        return np.column_stack([x["price"], x["title"].str.len(), rng.normal(size=(len(x), 38))])
    monkeypatch.setattr(compute, "embed_case_rows", embed)


def write_cache(x, path, monkeypatch):
    fake_case(monkeypatch)
    inputs, embeddings, _ = compute._encode_case_rows(x, case=None)
    write_embeddings(str(path), inputs=inputs, embeddings=embeddings, meta={"features": list(x.columns)},
                     dtype=np.float16)
    return read_embeddings(str(path))


def test_cache_keys_distinct_rows_and_reads_them_back(monkeypatch, tmp_path):
    x = pd.concat([table(), table().iloc[:3].set_axis([1, 2, 3])])
    cache = write_cache(x, tmp_path, monkeypatch)
    assert len(cache.inputs) == 40 and cache.embeddings.dtype == np.float16
    rows = case_rows(cache, x.iloc[[41, 0]])
    assert rows.dtype == np.float32 and np.allclose(rows[:, 0], x["price"].iloc[[1, 0]], atol=1e-2)


class Recorder:
    """Stands in for the wrapped model: keeps the input CASE gives it."""

    def __init__(self, **kwargs):
        self.kwargs, self.inputs = kwargs, []

    def fit(self, x, y):
        self.inputs.append(x)

    def predict(self, x):
        self.inputs.append(x)
        return np.zeros(len(x))


@pytest.fixture
def cached(monkeypatch, tmp_path):
    cache = write_cache(table(), tmp_path, monkeypatch)
    calls = []

    def cached_embeddings(encoder, dataset, size=None, fold=None):
        calls.append((encoder, dataset, size, fold))
        return cache
    monkeypatch.setattr(case_interface, "cached_embeddings", cached_embeddings)
    return calls


def wrap(model_cls, monkeypatch, **kwargs):
    monkeypatch.setattr(model_cls, "TABULAR_MODEL", Recorder)
    return model_cls(problem_type=SupervisedTask.BINARY, device="cpu", image_column=None, **kwargs)


def test_the_wrapped_model_gets_its_own_input_and_32_case_components(cached, monkeypatch):
    x, y = table(), Series(np.tile(["no", "yes"], 20), index=table().index)
    case = wrap(CaseKumoTabular, monkeypatch, dataset=TEXT, size=Size.TEN_K, fold=3, text_encoder=TextEncoder.E5_LARGE)
    case.fit(x.iloc[:36], y.iloc[:36])
    case.predict(x.iloc[36:])
    train, test = case.tabular_model.inputs
    assert list(train.columns) == list(x.columns) + [f"lm_pca_{i}" for i in range(32)]
    assert train[list(x.columns)].equals(x.iloc[:36]) and list(test.index) == list(x.index[36:])
    assert list(test.columns) == list(train.columns)
    assert case.tabular_model.kwargs["text_encoder"] == TextEncoder.E5_LARGE
    assert cached == [(Interface.CASE, TEXT, Size.TEN_K, 3)]


def test_fewer_components_than_32_for_fewer_rows(cached, monkeypatch):
    case = wrap(CaseKumoTabular, monkeypatch, dataset=TEXT, size=Size.TEN_K, fold=0)
    case.fit(table().iloc[:10], Series(np.tile([0, 1], 5), index=table().index[:10]))
    assert case.tabular_model.inputs[0].columns[-1] == "lm_pca_9"


def test_case_fits_the_real_model_with_the_text_embedded(cached, monkeypatch):
    x, y = table(), Series(np.tile(["no", "yes"], 20), index=table().index)
    case = CaseRandomForest(problem_type=SupervisedTask.BINARY, device="cpu", dataset=TEXT, size=Size.TEN_K, fold=0,
                            text_encoder=TextEncoder.TFIDF)
    case.fit(x.iloc[:30], y.iloc[:30])
    assert case.tabular_model.text_features == {"title"} and len(case.predict(x.iloc[30:])) == 10
    assert set(case.score_all_metrics(x.iloc[30:], y.iloc[30:])) >= {"test_score", "test_error"}


def test_case_needs_a_text_dataset_and_its_fold(cached):
    y = Series(np.tile([0, 1], 20), index=table().index)
    with pytest.raises(ValueError, match="text datasets"):
        CaseRandomForest(problem_type=SupervisedTask.BINARY, device="cpu", image_column="img", fold=0).fit(table(), y)
    with pytest.raises(ValueError, match="fold"):
        CaseRandomForest(problem_type=SupervisedTask.BINARY, device="cpu", dataset=TEXT).fit(table(), y)


def test_each_case_model_is_named_after_its_model_and_shares_its_venv():
    names = {m.TABULAR_MODEL.MODEL_NAME: (m.MODEL_NAME, m.SHORT_NAME) for m in CASE_MODELS}
    assert names["Kumo-Tabular"] == ("CASE-Kumo-Tabular", "casekumo") and len(CASE_MODELS) == 23
    assert all(m.ENV == m.TABULAR_MODEL.ENV and m.NEEDS_GPU == m.TABULAR_MODEL.NEEDS_GPU for m in CASE_MODELS)
    assert all(m.USE_TEXT_EMBEDDINGS for m in CASE_MODELS)
