"""The interfaces, by the name of their cached embeddings, and the model each one embeds with."""
from enum import StrEnum

from multabench.encoders.constants import Encoder
from multabench.interfaces.case.constants import CASE_MODEL
from multabench.interfaces.tile.constants import LLAMA_3_1_8B


class Interface(StrEnum):
    TILE = "tile"
    CASE = "case"


INTERFACES = {
    Interface.TILE: Encoder(encoder_name=LLAMA_3_1_8B, tune_encoder=False),
    # Not fine-tuned, but cached per fold like the tuned encoders, as its context is the fold's training rows.
    Interface.CASE: Encoder(encoder_name=CASE_MODEL, tune_encoder=True),
}
