"""Constants for E5 text encoders (BERT-based)."""

from enum import StrEnum

from multabench.utils.encoders import Encoder

E5_SMALL_V2 = "intfloat/e5-small-v2"
E5_LARGE_V2 = "intfloat/e5-large-v2"

TF_IDF = "tf-idf"  # skrub StringEncoder (TF-IDF + TruncatedSVD), CPU-only


class TextEncoder(StrEnum):
    TFIDF = "tfidf"
    E5_SMALL = "e5-small"
    E5_SMALL_TAR = "e5-small-tar"
    E5_LARGE = "e5-large"


TEXT_ENCODERS = {
    TextEncoder.TFIDF: Encoder(encoder_name=TF_IDF, tune_encoder=False),
    TextEncoder.E5_SMALL: Encoder(encoder_name=E5_SMALL_V2, tune_encoder=False),
    TextEncoder.E5_SMALL_TAR: Encoder(encoder_name=E5_SMALL_V2, tune_encoder=True),
    TextEncoder.E5_LARGE: Encoder(encoder_name=E5_LARGE_V2, tune_encoder=False),
}

# The modules LoRA adapts in each tuned BERT layer, encoder.layer.{i}.
LORA_TEXT_TARGET_MODULES = [
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
]

# E5 expects a task prefix for embedding; use "passage: " for generic text (e.g. when tuning on a text feature).
# When column name is provided, format is "passage: col_name: col_val".
E5_PASSAGE_PREFIX = "passage: "


def format_e5_passage(col_name: str, col_val: str) -> str:
    """Format text for E5 embedding: passage: col_name: col_val."""
    return f"{E5_PASSAGE_PREFIX}{col_name}: {col_val}"
