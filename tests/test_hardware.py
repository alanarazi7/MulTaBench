import json

import torch

from multabench.benchmark.runs import Run, has_result
from multabench.benchmark.sweep import pending_runs, write_jobs_file
from multabench.result_keys import OFFICIAL_HARDWARE
from multabench.utils import hardware
from multabench.utils.hardware import OFFICIAL_CPUS, OFFICIAL_RAM_GB, official_hardware_mismatches

RUN = Run("light", "BIN_TEXT_FAKE_JOB_POSTING", "tfidf", None, "10k", 0)


def test_cpu_is_not_the_official_hardware(monkeypatch):
    monkeypatch.setattr(hardware, "visible_cpus", lambda: OFFICIAL_CPUS)
    monkeypatch.setattr(hardware, "ram_limit_gb", lambda: float(OFFICIAL_RAM_GB))
    assert official_hardware_mismatches(torch.device("cpu")) == ["GPU is None, not NVIDIA RTX PRO 6000 Blackwell"]


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


def test_official_sweeps_redo_results_from_other_hardware(tmp_path):
    for official in (False, True):
        with open(tmp_path / f"{RUN.name}.json", "w") as f:
            json.dump({OFFICIAL_HARDWARE: official}, f)
        assert has_result(str(tmp_path / f"{RUN.name}.json"))
        assert has_result(str(tmp_path / f"{RUN.name}.json"), official=True) == official
        assert pending_runs([RUN], output_dir=str(tmp_path), official=True) == ([] if official else [RUN])


def test_official_flag_reaches_every_job(tmp_path):
    jobs = tmp_path / "jobs.txt"
    write_jobs_file([RUN], path=str(jobs), python="python", output_dir="runs", official=True)
    assert jobs.read_text().strip().endswith(" --official")
