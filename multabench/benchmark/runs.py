import shlex
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Run:
    """One benchmark.py run; an encoder is None when the dataset has no column of its kind."""
    model: str
    dataset: str
    text_encoder: Optional[str]
    image_encoder: Optional[str]
    size: str
    fold: int

    @property
    def name(self) -> str:
        return f"{self.model}_{self.dataset}_{self.text_encoder or 'none'}_{self.image_encoder or 'none'}_{self.size}_{self.fold}"

    def command(self, python: str, output_dir: str) -> str:
        args = [python, "benchmark.py", "--model", self.model, "--dataset_name", self.dataset,
                "--size", self.size, "--fold", str(self.fold), "--output_dir", output_dir]
        if self.text_encoder:
            args += ["--text_encoder", self.text_encoder]
        if self.image_encoder:
            args += ["--image_encoder", self.image_encoder]
        return shlex.join(args)
