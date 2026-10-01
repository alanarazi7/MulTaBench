"""CPU core count and torch device selection (copied from tabstar 1.1.15, tabstar/training/devices.py)."""
import os
from subprocess import PIPE, Popen
from typing import Optional

import torch

CPU_CORES = 8

EXISTING_CORES = os.cpu_count()
if EXISTING_CORES < CPU_CORES:
    print(f"❗ Warning: {CPU_CORES} CPU devices requested, but only {EXISTING_CORES} available.")
    CPU_CORES = EXISTING_CORES


def get_device(device: Optional[str | torch.device] = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    gpu = os.getenv('CUDA_VISIBLE_DEVICES')
    if gpu is not None:
        print(f"🖥️ Using CUDA_VISIBLE_DEVICES={gpu}. Setting device to 'cuda'.")
        if isinstance(gpu, int) or gpu.isdigit():
            return torch.device("cuda")
        raise ValueError(f"Invalid CUDA_VISIBLE_DEVICES value: {gpu}. It should be an integer for now.")
    if device is None:
        device = _get_device_type()
    return torch.device(device)


def _clear_cuda_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def _get_device_type() -> str:
    if torch.cuda.is_available():
        _clear_cuda_cache()
        return _get_free_gpu()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps"
    print(f"⚠️ No GPU available, using CPU. This may lead to slow performance.")
    return "cpu"


def _get_free_gpu() -> str:
    gpu_output = Popen(["nvidia-smi", "-q", "-d", "PIDS"], stdout=PIPE, encoding="utf-8")
    gpu_processes = Popen(["grep", "Processes"], stdin=gpu_output.stdout, stdout=PIPE, encoding="utf-8")
    gpu_output.stdout.close()
    processes_output = gpu_processes.communicate()[0]
    for i, line in enumerate(processes_output.strip().split("\n")):
        if line.endswith("None"):
            return f"cuda:{i}"
    raise RuntimeError("No free GPU found! However, pass `CUDA_VISIBLE_DEVICES` if you insist on using one.")
