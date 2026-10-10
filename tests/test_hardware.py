import pytest
import torch

from multabench.utils import hardware
from multabench.utils.hardware import (EMBEDDING_CPUS, EMBEDDING_GPU, OFFICIAL_CPUS, OFFICIAL_RAM_GB, assert_official_hardware,
                                       official_hardware_mismatches)


def test_cpu_is_not_the_official_hardware(monkeypatch):
    monkeypatch.setattr(hardware, "visible_cpus", lambda: OFFICIAL_CPUS)
    monkeypatch.setattr(hardware, "ram_limit_gb", lambda: float(OFFICIAL_RAM_GB))
    assert official_hardware_mismatches(torch.device("cpu")) == ["GPU is None, not NVIDIA RTX PRO 6000 Blackwell"]
    with pytest.raises(RuntimeError, match="Not the official hardware"):
        assert_official_hardware(torch.device("cpu"))


def test_the_embedding_cache_is_checked_against_its_own_hardware(monkeypatch):
    monkeypatch.setattr(hardware, "visible_cpus", lambda: EMBEDDING_CPUS)
    monkeypatch.setattr(hardware, "ram_limit_gb", lambda: float(OFFICIAL_RAM_GB))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA A100-SXM4-40GB")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    cuda = torch.device("cuda")
    assert official_hardware_mismatches(cuda, official_gpu=EMBEDDING_GPU, official_cpus=EMBEDDING_CPUS) == []
    assert official_hardware_mismatches(cuda) == ["GPU is NVIDIA A100-SXM4-40GB, not NVIDIA RTX PRO 6000 Blackwell",
                                                  "6 CPUs are available, not 8"]


def test_cpus_and_ram_must_match(monkeypatch):
    monkeypatch.setattr(hardware, "visible_cpus", lambda: 24)
    monkeypatch.setattr(hardware, "ram_limit_gb", lambda: 178.0)
    mismatches = official_hardware_mismatches(torch.device("cpu"))
    assert mismatches[1:] == ["24 CPUs are available, not 8", "RAM is limited to 178.0 GB, not 32 GB"]


def test_cgroup_v2_limit_on_an_ancestor_applies(tmp_path):
    job = tmp_path / "cgroup/slurm/job_1"
    (job / "step_0").mkdir(parents=True)
    (job / "memory.max").write_text(str(32 * 1024 ** 3))
    (job / "step_0/memory.max").write_text("max")
    (tmp_path / "proc").write_text("0::/slurm/job_1/step_0\n")
    assert hardware._cgroup_memory_limits(str(tmp_path / "proc"), str(tmp_path / "cgroup")) == [32 * 1024 ** 3]


def test_cgroup_v1_memory_controller(tmp_path):
    step = tmp_path / "cgroup/memory/slurm/job_1/step_0"
    step.mkdir(parents=True)
    (step / "memory.limit_in_bytes").write_text(str(16 * 1024 ** 3))
    (tmp_path / "proc").write_text("12:cpuset:/slurm/job_1/step_0\n4:memory:/slurm/job_1/step_0\n")
    assert hardware._cgroup_memory_limits(str(tmp_path / "proc"), str(tmp_path / "cgroup")) == [16 * 1024 ** 3]
