"""CPU core count and torch device selection."""
import os
from typing import Optional

import torch

CPU_CORES = 8

EXISTING_CORES = os.cpu_count()
if EXISTING_CORES < CPU_CORES:
    print(f"❗ Warning: {CPU_CORES} CPU devices requested, but only {EXISTING_CORES} available.")
    CPU_CORES = EXISTING_CORES


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print(f"⚠️ No GPU available, using CPU. This may lead to slow performance.")
    return torch.device("cpu")
