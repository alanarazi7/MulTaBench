import os
from dataclasses import dataclass
from os.path import dirname, join
from typing import Dict, Iterable, List, Optional

import psutil
import torch

from multabench.e5.constants import TF_IDF
from multabench.utils.devices import CPU_CORES
from multabench.utils.encoders import Encoder


@dataclass(frozen=True)
class Hardware:
    gpu_prefix: Optional[str]
    cpu_prefix: Optional[str]
    cpus: int
    ram_gb: int


# Measured times are only a proxy for a model's cost, and compare only between runs on the same spec. CPU runs use
# an Intel Xeon, since the GPU machines' AMD EPYC isn't available without a GPU.
GPU_RUN_HARDWARE = Hardware(gpu_prefix="NVIDIA RTX PRO 6000 Blackwell", cpu_prefix="AMD EPYC 9B45", cpus=8, ram_gb=32)
CPU_RUN_HARDWARE = Hardware(gpu_prefix=None, cpu_prefix="INTEL(R) XEON(R) PLATINUM 8581C", cpus=8, ram_gb=32)
EMBEDDING_HARDWARE = Hardware(gpu_prefix="NVIDIA A100", cpu_prefix=None, cpus=6, ram_gb=32)


def run_hardware(model_needs_gpu: bool, encoders: Iterable[Optional[Encoder]]) -> Hardware:
    """A run needs a GPU if its model does, or if it embeds text or images with a neural encoder."""
    encoder_needs_gpu = any(e is not None and e.encoder_name != TF_IDF for e in encoders)
    return GPU_RUN_HARDWARE if model_needs_gpu or encoder_needs_gpu else CPU_RUN_HARDWARE


def get_hardware_dict(device: torch.device) -> Dict:
    return {**_get_gpu_dict(device), **_get_cpu_dict(), "visible_cpus": visible_cpus(), "ram_limit_gb": ram_limit_gb()}


def assert_hardware(device: torch.device, expected: Hardware):
    mismatches = hardware_mismatches(device, expected)
    if mismatches:
        raise RuntimeError(f"Not the expected hardware: {'; '.join(mismatches)}")


def hardware_mismatches(device: torch.device, expected: Hardware) -> List[str]:
    mismatches = []
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    if expected.gpu_prefix is None:
        if device.type != "cpu":
            mismatches.append(f"Runs on {device}, not the CPU")
    elif gpu is None or not gpu.startswith(expected.gpu_prefix):
        mismatches.append(f"GPU is {gpu}, not {expected.gpu_prefix}")
    elif torch.cuda.device_count() != 1:
        mismatches.append(f"{torch.cuda.device_count()} GPUs are visible, not 1")
    cpu = get_cpu_name_linux()
    if expected.cpu_prefix is not None and (cpu is None or not cpu.startswith(expected.cpu_prefix)):
        mismatches.append(f"CPU is {cpu}, not {expected.cpu_prefix}")
    if visible_cpus() != expected.cpus:
        mismatches.append(f"{visible_cpus()} CPUs are available, not {expected.cpus}")
    if abs(ram_limit_gb() - expected.ram_gb) > 1:
        mismatches.append(f"RAM is limited to {ram_limit_gb():.1f} GB, not {expected.ram_gb} GB")
    return mismatches


def visible_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def ram_limit_gb() -> float:
    """The machine's RAM, or less if a cgroup (e.g. a Slurm job's --mem) limits this process."""
    return byte_to_gb(min([psutil.virtual_memory().total] + _cgroup_memory_limits()))


def _cgroup_memory_limits(proc_cgroup: str = "/proc/self/cgroup", cgroup_root: str = "/sys/fs/cgroup") -> List[int]:
    try:
        with open(proc_cgroup) as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    limits = []
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            root, limit_file = cgroup_root, "memory.max"
        elif "memory" in controllers.split(","):
            root, limit_file = join(cgroup_root, "memory"), "memory.limit_in_bytes"
        else:
            continue
        # A limit set on any ancestor cgroup applies too.
        while True:
            limit = _read_limit(join(root, path.lstrip("/"), limit_file))
            if limit is not None:
                limits.append(limit)
            if path in ("", "/"):
                break
            path = dirname(path)
    return limits


def _read_limit(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None
    return None if text == "max" else int(text)


def _get_gpu_dict(device: torch.device) -> Dict:
    use_gpu = False
    gpu_type = None
    gpu_total_mem_gb = None
    if device.type == 'cuda':
        use_gpu = True
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_type = torch.cuda.get_device_name(gpu_index)
        gpu_total_mem = torch.cuda.get_device_properties(gpu_index).total_memory
        gpu_total_mem_gb = byte_to_gb(gpu_total_mem)
    return {"gpu_type": gpu_type, "gpu_total_memory_gb": gpu_total_mem_gb, "use_gpu": use_gpu}


def _get_cpu_dict() -> Dict:
    ram_bytes = psutil.virtual_memory().total
    ram_gb = byte_to_gb(ram_bytes)
    cpu_name = get_cpu_name_linux()
    return {"cpu_cores": CPU_CORES, "system_ram_gb": ram_gb, "cpu_name": cpu_name}


def byte_to_gb(byte_size: int) -> float:
    return byte_size / (1024 ** 3)


def get_cpu_name_linux() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        return None
