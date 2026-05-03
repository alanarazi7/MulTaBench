"""Constants for E5 text encoders (BERT-based)."""

E5_SMALL_V2 = "intfloat/e5-small-v2"
E5_LARGE_V2 = "intfloat/e5-large-v2"

E5_SMALL = "e5-small"
E5_LARGE = "e5-large"
TF_IDF = "tf-idf"  # skrub StringEncoder (TF-IDF + TruncatedSVD), CPU-only

E5_MODEL_NAMES = {
    E5_SMALL: E5_SMALL_V2,
    E5_LARGE: E5_LARGE_V2,
}

D_E5_SMALL = 384
D_E5_LARGE = 1024

E5_SMALL_LAYERS = 12
E5_LARGE_LAYERS = 24

E5_DIM = {
    E5_SMALL_V2: D_E5_SMALL,
    E5_LARGE_V2: D_E5_LARGE,
}

E5_NUM_LAYERS = {
    E5_SMALL_V2: E5_SMALL_LAYERS,
    E5_LARGE_V2: E5_LARGE_LAYERS,
}

# BERT layer structure: encoder.layer.{i}.attention.self.{query,key,value}, attention.output.dense, intermediate.dense, output.dense
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
