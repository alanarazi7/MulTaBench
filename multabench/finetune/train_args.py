from dataclasses import dataclass, asdict


@dataclass
class DinoTrainArgs:
    lora_rank: int = 16
    img_layers: int = 3
    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 3
    weight_decay: float = 0.001
    batch_size: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class E5TrainArgs:
    lora_rank: int = 16
    text_layers: int = 3
    learning_rate: float = 1e-4
    epochs: int = 50
    patience: int = 3
    weight_decay: float = 0.001
    batch_size: int = 256

    def to_dict(self) -> dict:
        return asdict(self)
