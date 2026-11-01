"""The interfaces, by the name of their cached embeddings, and the model each one embeds with."""
from enum import StrEnum

from multabench.encoders.constants import Encoder
from multabench.interfaces.tile.constants import LLAMA_3_1_8B


class Interface(StrEnum):
    TILE = "tile"


INTERFACES = {
    Interface.TILE: Encoder(encoder_name=LLAMA_3_1_8B, tune_encoder=False),
}
