import os

from sklearn.decomposition import PCA

from tabstar.constants import SEED

from multabench.preprocessing.discretize import discretize_numerical

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppresses warning, avoids deadlock
from typing import Dict, Set, Optional, Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import torch

from multabench.e5.constants import E5_SMALL_V2, TF_IDF
from multabench.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5
from multabench.utils.pca_logging import log_pca_variance

PCA_COMPONENTS = 30


class _IdentityTransform:
    """No-op transformer: returns embeddings as-is (no PCA)."""
    def __init__(self, n_components: int):
        self.n_components = n_components

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class SkrubColumnEncoder:
    """Per-column text encoder using skrub.StringEncoder (TF-IDF + TruncatedSVD). CPU-only, no tokenizer needed."""

    def __init__(self, string_encoder: Any, col_name: str, n_components: int):
        self.string_encoder = string_encoder
        self.col_name = col_name
        self.n_components = n_components
        self.encoder = self  # so wrapper.encoder.transform(X) delegates to self.transform(X)

    def encode_texts(self, texts: list[str], device) -> np.ndarray:
        """Encode texts with skrub StringEncoder (device ignored — CPU-only)."""
        result = self.string_encoder.transform(pd.Series(texts, dtype=str))
        return np.asarray(result)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X  # encode_texts already returns final (N, n_components) array


class E5ColumnEncoder:
    """Per-column text encoder: holds E5 model, tokenizer (processor), and PCA (or identity). Uses passage: col_name: col_val format."""

    def __init__(self, model: Any, tokenizer: Any, encoder: Any, col_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.col_name = col_name
        self.n_components = encoder.n_components

    def encode_texts(self, texts: list[str], device: torch.device) -> np.ndarray:
        """Encode texts with this column's E5 model and tokenizer (passage: col_name: col_val)."""
        return encode_texts_with_e5(
            texts=texts,
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            col_name=self.col_name,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.encoder.transform(X)


def fit_text_encoders_skrub(
    x: DataFrame,
    text_features_list: list[str],
    pca_components: int = PCA_COMPONENTS,
) -> Dict[str, SkrubColumnEncoder]:
    """Fit one SkrubColumnEncoder per column using skrub.StringEncoder (TF-IDF + TruncatedSVD)."""
    from skrub import StringEncoder
    text_encoders: Dict[str, SkrubColumnEncoder] = {}
    for col in text_features_list:
        col_series = x[col].astype(str).fillna("")
        print(f"Fitting SkrubStringEncoder for column {col} with n_components={pca_components} for {len(col_series)} texts")
        string_enc = StringEncoder(n_components=pca_components)
        string_enc.fit(col_series)
        text_encoders[str(col)] = SkrubColumnEncoder(
            string_encoder=string_enc,
            col_name=str(col),
            n_components=pca_components,
        )
    return text_encoders


def fit_text_encoders_vanilla(
    x: DataFrame,
    text_features_list: list[str],
    device: torch.device,
    e5_model_name: str = E5_SMALL_V2,
    pca_components: int = PCA_COMPONENTS,
    no_pca: bool = False,
) -> Dict[str, E5ColumnEncoder]:
    """Fit one E5ColumnEncoder per column using shared vanilla E5 + PCA per column. Uses passage: col_name: col_val format."""
    text_encoders: Dict[str, E5ColumnEncoder] = {}
    model, tokenizer = get_vanilla_e5(device, model_name=e5_model_name)
    for col in text_features_list:
        texts = x[col].astype(str).fillna("").tolist()
        print(f"Fitting E5ColumnEncoder for column {col} with model {e5_model_name} for {len(texts)} texts")
        col_embeddings = encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer, device=device, col_name=str(col))
        if no_pca:
            encoder = _IdentityTransform(n_components=col_embeddings.shape[1])
        else:
            encoder = PCA(n_components=pca_components, random_state=SEED)
            encoder.fit(col_embeddings)
            log_pca_variance(pca=encoder, col_name=col)
        text_encoders[str(col)] = E5ColumnEncoder(model=model, tokenizer=tokenizer, encoder=encoder, col_name=str(col))
    return text_encoders


def fit_text_encoders_tuned(
    x: DataFrame,
    text_features_list: list[str],
    device: torch.device,
    y: Series,
    e5_train_kwargs: Dict[str, Any],
    is_cls: bool,
    d_output: int,
    e5_model_name: str = E5_SMALL_V2,
    pca_components: int = PCA_COMPONENTS,
    no_pca: bool = False,
) -> Dict[str, E5ColumnEncoder]:
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
    for col in text_features_list:
        texts = x[col].astype(str).fillna("").tolist()
        col_embeddings = encode_texts_with_e5(texts=texts, model=tuned_model, tokenizer=tuned_tokenizer, device=device, col_name=str(col))
        if no_pca:
            encoder = _IdentityTransform(n_components=col_embeddings.shape[1])
        else:
            encoder = PCA(n_components=pca_components, random_state=SEED)
            encoder.fit(col_embeddings)
            log_pca_variance(pca=encoder, col_name=col)
        text_encoders[col] = E5ColumnEncoder(model=tuned_model, tokenizer=tuned_tokenizer, encoder=encoder, col_name=str(col))
    return text_encoders


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
    pca_components: int = PCA_COMPONENTS,
    no_pca: bool = False,
) -> Dict[str, E5ColumnEncoder]:
    """
    Fit one E5 model per text column (or vanilla E5 shared across columns when not tuning).
    Each column gets an E5ColumnEncoder wrapper holding model, tokenizer, and PCA.
    Returns text_encoders mapping column -> E5ColumnEncoder.
    """
    text_features_list = sorted(text_features)
    if not text_features_list:
        return {}
    if e5_model_name == TF_IDF:
        return fit_text_encoders_skrub(
            x=x,
            text_features_list=text_features_list,
            pca_components=pca_components,
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
            pca_components=pca_components,
            no_pca=no_pca,
        )
    return fit_text_encoders_vanilla(
        x=x,
        text_features_list=text_features_list,
        device=device,
        e5_model_name=e5_model_name,
        pca_components=pca_components,
        no_pca=no_pca,
    )


def transform_text_features(
    x: DataFrame,
    text_encoders: Dict[str, E5ColumnEncoder],
    device: torch.device,
) -> DataFrame:
    for text_col, wrapper in text_encoders.items():
        texts = x[text_col].astype(str).fillna("").tolist()
        embeddings = wrapper.encode_texts(texts, device)
        n_components = wrapper.n_components
        pca_vec = wrapper.encoder.transform(embeddings)
        pca_cols = [f"{text_col}_txt_pca_{i}" for i in range(n_components)]
        pca_df = pd.DataFrame(pca_vec, index=x.index, columns=pca_cols)
        cols_before = len(x.columns)
        x = x.drop(columns=[text_col])
        x = pd.concat([x, pca_df], axis=1)
        assert len(x.columns) == cols_before + n_components - 1
    return x
