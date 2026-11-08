"""TabICL's inference, which TabLDM ships a copy of, keeps its largest tensor, the column embeddings, on the GPU if it
fits, else in CPU RAM, else on disk. It sizes CPU RAM by the machine's free RAM, not a job's limit (e.g. Slurm's --mem),
so a run with many features can fill the limit and be killed, with no error or result."""
import tempfile
from typing import Dict

from pandas import DataFrame

from multabench.utils.hardware import available_ram_mb

ONE_MEMBER_CELLS = 10_000_000


def offload_within_ram_limit(inference_manager: type) -> Dict[str, str]:
    """Sends the tensor to disk where the engine picks CPU RAM but the tensor alone exceeds what the job has left, a run
    that would be killed; every other decision is the engine's own. Returns the disk to offload to, as model kwargs."""
    if not hasattr(inference_manager, "resolve_without_ram_limit"):
        inference_manager.resolve_without_ram_limit = inference_manager._resolve_offload_mode
        inference_manager._resolve_offload_mode = _resolve_within_ram_limit
    return {"disk_offload_dir": tempfile.gettempdir()}


def _resolve_within_ram_limit(self, output_mb: float, gpu_free_mb: float, cpu_free_mb: float, disk_free_mb: float):
    mode, reason = self.resolve_without_ram_limit(output_mb, gpu_free_mb, cpu_free_mb, disk_free_mb)
    left_mb = available_ram_mb()
    if mode.name == "CPU" and output_mb > left_mb:
        detail = f"{output_mb:.0f}MB > {left_mb:.0f}MB left in the job's RAM -> disk"
        return type(mode).DISK, type(reason)("job_ram_limit", detail)
    return mode, reason


def one_member_above_10m_cells(x_train: DataFrame) -> Dict[str, int]:
    """One ensemble member per forward pass above 10M training cells, as TabLDM's "auto" does, not changing results."""
    return {"batch_size": 1} if x_train.size > ONE_MEMBER_CELLS else {}
