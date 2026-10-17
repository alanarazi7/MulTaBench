from dataclasses import asdict, dataclass

REGRESSION_BINS = 20
DATALOADER_WORKERS = 4


@dataclass(frozen=True)
class LoraArgs:
    learning_rate: float
    epochs: int
    lora_rank: int = 16
    layers: int = 3
    patience: int = 3
    weight_decay: float = 0.001
    batch_size: int = 256

    def to_dict(self) -> dict:
        return asdict(self)

