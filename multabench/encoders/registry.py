"""Each text and image encoder a run can choose, by name, to the class that embeds its columns."""
from typing import Dict, Type

from multabench.embeddings.hub import CachedEncoder
from multabench.encoders.abstract_encoder import EmbeddingEncoder
from multabench.encoders.constants import ImageEncoder, TextEncoder
from multabench.encoders.dino.dino_encoder import DinoEncoder
from multabench.encoders.e5.e5_encoder import E5Encoder
from multabench.encoders.tfidf import TfidfEncoder
from multabench.encoders.tile.tile_encoder import TileEncoder

ENCODER_CLASSES: Dict[CachedEncoder, Type[EmbeddingEncoder]] = {
    TextEncoder.TFIDF: TfidfEncoder,
    TextEncoder.E5_SMALL: E5Encoder,
    TextEncoder.E5_SMALL_TAR: E5Encoder,
    TextEncoder.E5_LARGE: E5Encoder,
    TextEncoder.E5_LARGE_TAR: E5Encoder,
    TextEncoder.TILE: TileEncoder,
    ImageEncoder.DINO_SMALL: DinoEncoder,
    ImageEncoder.DINO_SMALL_TAR: DinoEncoder,
    ImageEncoder.DINO_LARGE: DinoEncoder,
    ImageEncoder.DINO_LARGE_TAR: DinoEncoder,
}


def make_encoder(name: CachedEncoder, **kwargs) -> EmbeddingEncoder:
    return ENCODER_CLASSES[name](name, **kwargs)
