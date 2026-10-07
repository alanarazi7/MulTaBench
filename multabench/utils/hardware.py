import os
from os.path import dirname, join
from typing import Dict, List, Optional

import psutil
import torch

from multabench.utils.devices import CPU_CORES


# Leaderboard runs all use this machine, so their times compare: TabArena's 8 CPU cores and 32 GB of RAM, with one
# NVIDIA L4 (24 GB) where TabArena used an L40S (48 GB).
OFFICIAL_GPU = "NVIDIA L4"
OFFICIAL_CPUS = 8
OFFICIAL_RAM_GB = 32


def get_hardware_dict(device: torch.device) -> Dict:
    return {**_get_gpu_dict(device), **_get_cpu_dict(), "visible_cpus": visible_cpus(), "ram_limit_gb": ram_limit_gb()}


def official_hardware_mismatches(device: torch.device) -> List[str]:
    """How this process's hardware differs from the official one; empty when it matches."""
    mismatches = []
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    if gpu != OFFICIAL_GPU:
        mismatches.append(f"GPU is {gpu}, not {OFFICIAL_GPU}")
    elif torch.cuda.device_count() != 1:
        mismatches.append(f"{torch.cuda.device_count()} GPUs are visible, not 1")
    if visible_cpus() != OFFICIAL_CPUS:
        mismatches.append(f"{visible_cpus()} CPUs are available, not {OFFICIAL_CPUS}")
    if abs(ram_limit_gb() - OFFICIAL_RAM_GB) > 1:
        mismatches.append(f"RAM is limited to {ram_limit_gb():.1f} GB, not {OFFICIAL_RAM_GB} GB")
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
