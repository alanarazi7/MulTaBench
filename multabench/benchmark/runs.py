import shlex
from dataclasses import dataclass
from typing import Optional

from multabench.benchmark.splits import Size
from multabench.dino.constants import ImageEncoder
from multabench.e5.constants import TextEncoder


@dataclass(frozen=True)
class Run:
    """One benchmark.py run; an encoder is None if the dataset lacks its columns or the model reads them raw."""
    model: str
    dataset: str
    text_encoder: Optional[TextEncoder]
    image_encoder: Optional[ImageEncoder]
    size: Size
    fold: int

    @property
    def name(self) -> str:
        txt = self.text_encoder or "none"
        img = self.image_encoder or "none"
        return f"{self.model}_{self.dataset}_{txt}_{img}_{self.size}_{self.fold}"

    def command(self, python: str, output_dir: str, official: bool = False, overwrite: bool = False) -> str:
        args = [python, "benchmark.py", "--model", self.model, "--dataset_name", self.dataset,
                "--size", self.size, "--fold", str(self.fold), "--output_dir", output_dir]
        if official:
            args.append("--official")
        if overwrite:
            args.append("--overwrite")
        if self.text_encoder:
            args += ["--text_encoder", self.text_encoder]
        if self.image_encoder:
            args += ["--image_encoder", self.image_encoder]
        return shlex.join(args)
