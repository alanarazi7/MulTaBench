import os

from sklearn.decomposition import PCA

from multabench.constants import SEED

from multabench.preprocessing.discretize import discretize_numerical

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppresses warning, avoids deadlock
from typing import Dict, Set, Optional, Any, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import torch

from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.e5.constants import E5_SMALL_V2, TF_IDF
from multabench.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5
from multabench.embeddings.hub import cached_embeddings, cached_encoder, select_rows
from multabench.utils.timing import embedding_step

PCA_COMPONENTS = 30


class SkrubColumnEncoder:
    """Per-column text encoder using skrub.StringEncoder (TF-IDF + TruncatedSVD). CPU-only, no tokenizer needed."""

    def __init__(self, string_encoder: Any, col_name: str, n_components: int):
        self.string_encoder = string_encoder
        self.col_name = col_name
        self.n_components = n_components
        self.encoder = self  # so wrapper.encoder.transform(X) delegates to self.transform(X)

    @embedding_step
    def encode_texts(self, texts: list[str], device) -> np.ndarray:
        """Encode texts with skrub StringEncoder (device ignored — CPU-only)."""
        result = self.string_encoder.transform(pd.Series(texts, dtype=str))
        return np.asarray(result)

    def embed(self, x: DataFrame, device) -> np.ndarray:
        return self.encode_texts(x[self.col_name].astype(str).fillna("").tolist(), device)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X  # encode_texts already returns final (N, n_components) array


class E5ColumnEncoder:
    """Per-column text encoder: holds E5 model, tokenizer (processor), and PCA (or identity). Uses passage: col_name: col_val format."""

    def __init__(self, model: Any, tokenizer: Any, encoder: Any, col_name: str,
                 cached: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.col_name = col_name
        self.n_components = encoder.n_components
        self.cached = cached

    def encode_texts(self, texts: list[str], device: torch.device) -> np.ndarray:
        """Encode texts with this column's E5 model and tokenizer (passage: col_name: col_val)."""
        return encode_texts_with_e5(
            texts=texts,
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            col_name=self.col_name,
        )

    def embed(self, x: DataFrame, device: torch.device) -> np.ndarray:
        """The embeddings of x's rows, from the cache when there is one: x's index holds the rows' positions in the dataset."""
        if self.cached is None:
            return self.encode_texts(x[self.col_name].astype(str).fillna("").tolist(), device)
        rows, array = self.cached
        return select_rows(rows, {self.col_name: array}, x.index.to_numpy())[self.col_name]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.encoder.transform(X)


@embedding_step
def fit_text_encoders_skrub(
    x: DataFrame,
    text_features_list: list[str],
) -> Tuple[Dict[str, SkrubColumnEncoder], Dict[str, np.ndarray]]:
    """Fit one SkrubColumnEncoder per column using skrub.StringEncoder (TF-IDF + TruncatedSVD)."""
    from skrub import StringEncoder
    text_encoders: Dict[str, SkrubColumnEncoder] = {}
    for col in text_features_list:
        col_series = x[col].astype(str).fillna("")
        print(f"Fitting SkrubStringEncoder for column {col} with n_components={PCA_COMPONENTS} for {len(col_series)} texts")
        string_enc = StringEncoder(n_components=PCA_COMPONENTS)
        string_enc.fit(col_series)
        text_encoders[str(col)] = SkrubColumnEncoder(
            string_encoder=string_enc,
            col_name=str(col),
            n_components=PCA_COMPONENTS,
        )
    return text_encoders, {}


def fit_text_encoders_vanilla(
    x: DataFrame,
    text_features_list: list[str],
    device: torch.device,
    e5_model_name: str = E5_SMALL_V2,
    dataset: Optional[MulTaBenchDatasetID] = None,
) -> Tuple[Dict[str, E5ColumnEncoder], Dict[str, np.ndarray]]:
    """Fit one E5ColumnEncoder per column using shared vanilla E5 + PCA per column. Uses passage: col_name: col_val format.
    Columns with cached embeddings on the Hub are read from there instead of being encoded."""
    text_encoders: Dict[str, E5ColumnEncoder] = {}
    train_embeddings: Dict[str, np.ndarray] = {}
    encoder_id = cached_encoder(e5_model_name, tuned=False)
    rows, cached = (cached_embeddings(encoder_id, dataset) if dataset is not None and encoder_id else None) or (None, {})
    model, tokenizer = None, None
    for col in text_features_list:
        col_cache = (rows, cached[col]) if col in cached else None
        if col_cache is None and model is None:
            model, tokenizer = get_vanilla_e5(device, model_name=e5_model_name)
        source = f"the cached {encoder_id} embeddings" if col_cache else f"model {e5_model_name}"
        print(f"Fitting E5ColumnEncoder for column {col} with {source} for {len(x)} texts")
        wrapper = E5ColumnEncoder(model=model, tokenizer=tokenizer, encoder=PCA(n_components=PCA_COMPONENTS, random_state=SEED),
                                  col_name=str(col), cached=col_cache)
        col_embeddings = wrapper.embed(x, device)
        wrapper.encoder.fit(col_embeddings)
        text_encoders[str(col)] = wrapper
        train_embeddings[str(col)] = col_embeddings
    return text_encoders, train_embeddings


def fit_text_encoders_tuned(
    x: DataFrame,
    text_features_list: list[str],
    device: torch.device,
    y: Series,
    e5_train_kwargs: Dict[str, Any],
    is_cls: bool,
    d_output: int,
    e5_model_name: str = E5_SMALL_V2,
) -> Tuple[Dict[str, E5ColumnEncoder], Dict[str, np.ndarray]]:
    """Fit a single E5 model for all text columns with passage: col_name: col_val format. Each column gets an E5ColumnEncoder sharing the same tuned model."""
    from transformers import AutoTokenizer

    from multabench.e5.e5_finetune import finetune_e5_with_lora
    from multabench.preprocessing.splits import split_to_val

    if not is_cls:
        y = discretize_numerical(y, n_bins=20)
    x_tr, x_val, y_tr, y_val = split_to_val(x=x, y=y, is_cls=True)
    kwargs = e5_train_kwargs or {}

    # Build combined train/val: for each row, for each text col, create (col_name: col_val, label)
    train_texts: list[str] = []
    train_y: list = []
    for idx in x_tr.index:
        row_y = y_tr.loc[idx]
        for col in text_features_list:
            val = str(x_tr.loc[idx, col]).strip() if pd.notna(x_tr.loc[idx, col]) else ""
            train_texts.append(f"{col}: {val}")
            train_y.append(row_y)

    val_texts: list[str] = []
    val_y: list = []
    for idx in x_val.index:
        row_y = y_val.loc[idx]
        for col in text_features_list:
            val = str(x_val.loc[idx, col]).strip() if pd.notna(x_val.loc[idx, col]) else ""
            val_texts.append(f"{col}: {val}")
            val_y.append(row_y)

    train_y_arr = np.array(train_y)
    val_y_arr = np.array(val_y)

    tokenizer = AutoTokenizer.from_pretrained(e5_model_name)
    print(f"Finetuning single E5 ({e5_model_name}) for {len(text_features_list)} columns with {len(train_texts)} train examples (passage: col_name: col_val)")
    tuned_model, tuned_tokenizer = finetune_e5_with_lora(
        train_texts=train_texts,
        train_y=train_y_arr,
        val_texts=val_texts,
        val_y=val_y_arr,
        device=device,
        tokenizer=tokenizer,
        model_name=e5_model_name,
        **kwargs,
    )
    tuned_model.to(device)

    text_encoders: Dict[str, E5ColumnEncoder] = {}
    train_embeddings: Dict[str, np.ndarray] = {}
    for col in text_features_list:
        texts = x[col].astype(str).fillna("").tolist()
        col_embeddings = encode_texts_with_e5(texts=texts, model=tuned_model, tokenizer=tuned_tokenizer, device=device, col_name=str(col))
        encoder = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        encoder.fit(col_embeddings)
        text_encoders[col] = E5ColumnEncoder(model=tuned_model, tokenizer=tuned_tokenizer, encoder=encoder, col_name=str(col))
        train_embeddings[col] = col_embeddings
    return text_encoders, train_embeddings


def fit_text_encoders(
    x: DataFrame,
    text_features: Set[str],
    device: torch.device,
    y: Optional[Series] = None,
    tune_e5: bool = False,
    e5_train_kwargs: Optional[Dict[str, Any]] = None,
    is_cls: bool = True,
    d_output: int = 2,
    e5_model_name: str = E5_SMALL_V2,
    dataset: Optional[MulTaBenchDatasetID] = None,
) -> Tuple[Dict[str, E5ColumnEncoder], Dict[str, np.ndarray]]:
    """
    Fit one E5 model per text column (or vanilla E5 shared across columns when not tuning).
    Each column gets an E5ColumnEncoder wrapper holding model, tokenizer, and PCA.
    Returns text_encoders mapping column -> E5ColumnEncoder, and the training rows' embeddings per column.
    """
    text_features_list = sorted(text_features)
    if not text_features_list:
        return {}, {}
    if e5_model_name == TF_IDF:
        return fit_text_encoders_skrub(
            x=x,
            text_features_list=text_features_list,
        )
    if tune_e5 and y is not None:
        return fit_text_encoders_tuned(
            x=x,
            text_features_list=text_features_list,
            device=device,
            y=y,
            e5_train_kwargs=e5_train_kwargs or {},
            is_cls=is_cls,
            d_output=d_output,
            e5_model_name=e5_model_name,
        )
    return fit_text_encoders_vanilla(
        x=x,
        text_features_list=text_features_list,
        device=device,
        e5_model_name=e5_model_name,
        dataset=dataset,
    )


def transform_text_features(
    x: DataFrame,
    text_encoders: Dict[str, E5ColumnEncoder],
    device: torch.device,
    train_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> DataFrame:
    train_embeddings = train_embeddings or {}
    for text_col, wrapper in text_encoders.items():
        embeddings = train_embeddings.get(text_col)
        if embeddings is None:
            embeddings = wrapper.embed(x, device)
        assert len(embeddings) == len(x)
        n_components = wrapper.n_components
        pca_vec = wrapper.encoder.transform(embeddings)
        pca_cols = [f"{text_col}_txt_pca_{i}" for i in range(n_components)]
        pca_df = pd.DataFrame(pca_vec, index=x.index, columns=pca_cols)
        cols_before = len(x.columns)
        x = x.drop(columns=[text_col])
        x = pd.concat([x, pca_df], axis=1)
        assert len(x.columns) == cols_before + n_components - 1
    return x
