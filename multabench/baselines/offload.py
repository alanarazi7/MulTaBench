"""TabICL's inference, which TabLDM ships a copy of, keeps its largest tensor, the column embeddings, on the GPU if it
fits, else in CPU RAM, else on disk. It sizes CPU RAM by the machine's free RAM, not a job's limit (e.g. Slurm's --mem),
so a run with many features can fill the limit and be killed, with no error or result."""
import tempfile
from typing import Dict

from multabench.utils.hardware import available_ram_mb


def offload_within_ram_limit(inference_manager: type) -> Dict[str, str]:
    """Makes the InferenceManager class size CPU RAM by the limit; returns the disk to fall back on, as model kwargs."""
    inference_manager.get_available_cpu_memory = lambda self: available_ram_mb()
    return {"disk_offload_dir": tempfile.gettempdir()}
