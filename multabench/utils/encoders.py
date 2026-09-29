from typing import NamedTuple


class Encoder(NamedTuple):
    encoder_name: str
    tune_encoder: bool  # fine-tune it on the task, i.e. target-aware "TAR"
