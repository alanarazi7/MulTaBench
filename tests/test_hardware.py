import pytest
import torch

from multabench.utils import hardware
from multabench.e5.constants import TEXT_ENCODERS, TextEncoder
from multabench.dino.constants import IMAGE_ENCODERS, ImageEncoder
from multabench.utils.hardware import (CPU_RUN_HARDWARE, EMBEDDING_HARDWARE, GPU_RUN_HARDWARE, Hardware, assert_hardware,
                                       hardware_mismatches, run_hardware)


def _machine(monkeypatch, hw: Hardware, gpu_name=None):
    monkeypatch.setattr(hardware, "get_cpu_name_linux", lambda: f"{hw.cpu_prefix} CPU @ 2.30GHz")
    monkeypatch.setattr(hardware, "visible_cpus", lambda: hw.cpus)
    monkeypatch.setattr(hardware, "ram_limit_gb", lambda: float(hw.ram_gb))
    if gpu_name:
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: gpu_name)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


def test_cpu_is_not_the_benchmark_hardware(monkeypatch):
    _machine(monkeypatch, GPU_RUN_HARDWARE)
    assert hardware_mismatches(torch.device("cpu"), GPU_RUN_HARDWARE) == ["GPU is None, not NVIDIA RTX PRO 6000 Blackwell"]
    with pytest.raises(RuntimeError, match="Not the expected hardware"):
        assert_hardware(torch.device("cpu"), GPU_RUN_HARDWARE)


def test_the_embedding_cache_is_checked_against_its_own_hardware(monkeypatch):
    _machine(monkeypatch, EMBEDDING_HARDWARE, gpu_name="NVIDIA A100-SXM4-40GB")
    cuda = torch.device("cuda")
    assert hardware_mismatches(cuda, EMBEDDING_HARDWARE) == []
    assert hardware_mismatches(cuda, GPU_RUN_HARDWARE)[0] == "GPU is NVIDIA A100-SXM4-40GB, not NVIDIA RTX PRO 6000 Blackwell"
    assert "6 CPUs are available, not 8" in hardware_mismatches(cuda, GPU_RUN_HARDWARE)


def test_cpu_hardware_runs_on_the_cpu(monkeypatch):
    _machine(monkeypatch, CPU_RUN_HARDWARE, gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition")
    assert hardware_mismatches(torch.device("cpu"), CPU_RUN_HARDWARE) == []
    assert hardware_mismatches(torch.device("cuda"), CPU_RUN_HARDWARE) == ["Runs on cuda, not the CPU"]
    assert hardware_mismatches(torch.device("cuda"), GPU_RUN_HARDWARE) == [
        "CPU is INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, not AMD EPYC 9B45"]


def test_runs_need_a_gpu_for_the_model_or_a_neural_encoder():
    tfidf, e5, dino = TEXT_ENCODERS[TextEncoder.TFIDF], TEXT_ENCODERS[TextEncoder.E5_SMALL], IMAGE_ENCODERS[ImageEncoder.DINO_SMALL]
    assert run_hardware(model_needs_gpu=False, encoders=[tfidf, None]) == CPU_RUN_HARDWARE
    assert run_hardware(model_needs_gpu=False, encoders=[None, None]) == CPU_RUN_HARDWARE
    assert run_hardware(model_needs_gpu=False, encoders=[e5, None]) == GPU_RUN_HARDWARE
    assert run_hardware(model_needs_gpu=False, encoders=[tfidf, dino]) == GPU_RUN_HARDWARE
    assert run_hardware(model_needs_gpu=True, encoders=[tfidf, None]) == GPU_RUN_HARDWARE


def test_cpus_and_ram_must_match(monkeypatch):
    _machine(monkeypatch, GPU_RUN_HARDWARE)
    monkeypatch.setattr(hardware, "visible_cpus", lambda: 24)
    monkeypatch.setattr(hardware, "ram_limit_gb", lambda: 178.0)
    mismatches = hardware_mismatches(torch.device("cpu"), GPU_RUN_HARDWARE)
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
