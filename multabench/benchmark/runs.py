import json
import os
import shlex
from dataclasses import dataclass
from typing import Optional

from multabench.dino.constants import ImageEncoder
from multabench.e5.constants import TextEncoder
from multabench.result_keys import OFFICIAL_HARDWARE


@dataclass(frozen=True)
class Run:
    """One benchmark.py run; an encoder is None when the dataset has no column of its kind."""
    model: str
    dataset: str
    text_encoder: Optional[TextEncoder]
    image_encoder: Optional[ImageEncoder]
    size: str
    fold: int

    @property
    def name(self) -> str:
        txt = self.text_encoder or "none"
        img = self.image_encoder or "none"
        return f"{self.model}_{self.dataset}_{txt}_{img}_{self.size}_{self.fold}"

    def command(self, python: str, output_dir: str) -> str:
        args = [python, "benchmark.py", "--model", self.model, "--dataset_name", self.dataset,
                "--size", self.size, "--fold", str(self.fold), "--output_dir", output_dir]
        if self.text_encoder:
            args += ["--text_encoder", self.text_encoder]
        if self.image_encoder:
            args += ["--image_encoder", self.image_encoder]
        return shlex.join(args)


def has_result(path: str, official: bool = False) -> bool:
    """With official, a result from other hardware doesn't count, so the run is redone on the official one."""
    if not os.path.exists(path):
        return False
    if not official:
        return True
    with open(path) as f:
        return json.load(f).get(OFFICIAL_HARDWARE) is True
