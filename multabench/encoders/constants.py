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
    E5_SMALL_LIGHTTAR = "e5-small-lighttar"
    E5_LARGE_LIGHTTAR = "e5-large-lighttar"


class ImageEncoder(StrEnum):
    DINO_SMALL = "dino-small"
    DINO_SMALL_TAR = "dino-small-tar"
    DINO_LARGE = "dino-large"
    DINO_LARGE_TAR = "dino-large-tar"
    DINO_SMALL_LIGHTTAR = "dino-small-lighttar"
    DINO_LARGE_LIGHTTAR = "dino-large-lighttar"


DEFAULT_TEXT_ENCODER = TextEncoder.E5_SMALL
DEFAULT_IMAGE_ENCODER = ImageEncoder.DINO_SMALL

TEXT_ENCODERS = {
    TextEncoder.TFIDF: Encoder(encoder_name=TF_IDF, tune_encoder=False),
    TextEncoder.E5_SMALL: Encoder(encoder_name=E5_SMALL_V2, tune_encoder=False),
    TextEncoder.E5_SMALL_TAR: Encoder(encoder_name=E5_SMALL_V2, tune_encoder=True),
    TextEncoder.E5_LARGE: Encoder(encoder_name=E5_LARGE_V2, tune_encoder=False),
    TextEncoder.E5_LARGE_TAR: Encoder(encoder_name=E5_LARGE_V2, tune_encoder=True),
    TextEncoder.E5_SMALL_LIGHTTAR: Encoder(encoder_name=E5_SMALL_V2, tune_encoder=False),
    TextEncoder.E5_LARGE_LIGHTTAR: Encoder(encoder_name=E5_LARGE_V2, tune_encoder=False),
}
# LightTAR reads the last layer from its frozen encoder's cache; its own cache has only the middle layer.
LIGHTTAR_LAST_LAYER = {TextEncoder.E5_SMALL_LIGHTTAR: TextEncoder.E5_SMALL,
                       TextEncoder.E5_LARGE_LIGHTTAR: TextEncoder.E5_LARGE}

IMAGE_ENCODERS = {
    ImageEncoder.DINO_SMALL: Encoder(encoder_name=DINOV3_SMALL, tune_encoder=False),
    ImageEncoder.DINO_SMALL_TAR: Encoder(encoder_name=DINOV3_SMALL, tune_encoder=True),
    ImageEncoder.DINO_LARGE: Encoder(encoder_name=DINOV3_LARGE, tune_encoder=False),
    ImageEncoder.DINO_LARGE_TAR: Encoder(encoder_name=DINOV3_LARGE, tune_encoder=True),
    ImageEncoder.DINO_SMALL_LIGHTTAR: Encoder(encoder_name=DINOV3_SMALL, tune_encoder=False),
    ImageEncoder.DINO_LARGE_LIGHTTAR: Encoder(encoder_name=DINOV3_LARGE, tune_encoder=False),
}
# DINO's frozen cache has its last layer after the final norm, so LightTAR's image cache has both of its layers.
LIGHTTAR_IMAGE_ENCODERS = (ImageEncoder.DINO_SMALL_LIGHTTAR, ImageEncoder.DINO_LARGE_LIGHTTAR)
