from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from multabench.baselines import offload
from multabench.baselines.offload import offload_within_ram_limit


class Mode(Enum):
    GPU = auto()
    CPU = auto()
    DISK = auto()


@dataclass
class Reason:
    key: str
    detail: Optional[str] = None


def manager(mode: Mode) -> type:
    """A stand-in for TabICL's InferenceManager, whose own resolution picks mode."""
    return type("InferenceManager", (), {"_resolve_offload_mode": lambda self, *sizes: (mode, Reason("auto"))})


def test_cpu_offload_goes_to_disk_only_when_the_tensor_exceeds_the_ram_left(monkeypatch):
    monkeypatch.setattr(offload, "available_ram_mb", lambda: 24_000)
    cls = manager(Mode.CPU)
    assert offload_within_ram_limit(cls)["disk_offload_dir"]
    assert cls()._resolve_offload_mode(50_000, 0, 200_000, 100_000)[0] == Mode.DISK
    assert cls()._resolve_offload_mode(20_000, 0, 200_000, 100_000) == (Mode.CPU, Reason("auto"))


def test_other_decisions_are_the_engines_and_patching_twice_wraps_once(monkeypatch):
    monkeypatch.setattr(offload, "available_ram_mb", lambda: 24_000)
    cls = manager(Mode.GPU)
    offload_within_ram_limit(cls)
    offload_within_ram_limit(cls)
    assert cls()._resolve_offload_mode(50_000, 80_000, 200_000, 100_000) == (Mode.GPU, Reason("auto"))
    assert cls.resolve_without_ram_limit is not offload._resolve_within_ram_limit
