"""The text and image encoders a run can choose, and the model each one uses."""
from enum import StrEnum
from typing import NamedTuple

from multabench.encoders.dino.constants import DINOV3_LARGE, DINOV3_SMALL
from multabench.encoders.e5.constants import E5_LARGE_V2, E5_SMALL_V2

TF_IDF = "tf-idf"  # skrub StringEncoder (TF-IDF + TruncatedSVD), CPU-only


class Encoder(NamedTuple):
    encoder_name: str
    tune_encoder: bool


class TextEncoder(StrEnum):
    TFIDF = "tfidf"
    E5_SMALL = "e5-small"
    E5_SMALL_TAR = "e5-small-tar"
    E5_LARGE = "e5-large"
    E5_LARGE_TAR = "e5-large-tar"


class ImageEncoder(StrEnum):
    DINO_SMALL = "dino-small"
    DINO_SMALL_TAR = "dino-small-tar"
    DINO_LARGE = "dino-large"
    DINO_LARGE_TAR = "dino-large-tar"


DEFAULT_TEXT_ENCODER = TextEncoder.E5_SMALL
DEFAULT_IMAGE_ENCODER = ImageEncoder.DINO_SMALL

TEXT_ENCODERS = {
    TextEncoder.TFIDF: Encoder(encoder_name=TF_IDF, tune_encoder=False),
    TextEncoder.E5_SMALL: Encoder(encoder_name=E5_SMALL_V2, tune_encoder=False),
    TextEncoder.E5_SMALL_TAR: Encoder(encoder_name=E5_SMALL_V2, tune_encoder=True),
    TextEncoder.E5_LARGE: Encoder(encoder_name=E5_LARGE_V2, tune_encoder=False),
    TextEncoder.E5_LARGE_TAR: Encoder(encoder_name=E5_LARGE_V2, tune_encoder=True),
}

IMAGE_ENCODERS = {
    ImageEncoder.DINO_SMALL: Encoder(encoder_name=DINOV3_SMALL, tune_encoder=False),
    ImageEncoder.DINO_SMALL_TAR: Encoder(encoder_name=DINOV3_SMALL, tune_encoder=True),
    ImageEncoder.DINO_LARGE: Encoder(encoder_name=DINOV3_LARGE, tune_encoder=False),
    ImageEncoder.DINO_LARGE_TAR: Encoder(encoder_name=DINOV3_LARGE, tune_encoder=True),
}
