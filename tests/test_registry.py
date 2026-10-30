import os
import subprocess
import sys

import pytest

REPO_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PACKAGES = ["autogluon.multimodal", "catboost", "causilo", "exaonetabular", "inference", "lightgbm", "pytabkit",
                  "sap_rpt_oss", "sdm", "tabarena", "tabdpt", "tabfm", "tabicl", "tabldm", "tabpfn", "tabpfn_extensions",
                  "tabstar", "xgboost"]
BLOCK_AND_IMPORT = f"""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if any(name == p or name.startswith(p + ".") for p in {MODEL_PACKAGES}):
            raise ImportError(name)
sys.meta_path.insert(0, Block())
import benchmark, sweep
print(len(benchmark.BASELINES))
"""


def test_every_wrapper_imports_without_its_model_package():
    done = subprocess.run([sys.executable, "-c", BLOCK_AND_IMPORT], cwd=REPO_DIR, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_a_run_without_its_required_token_stops_before_writing_a_result(tmp_path):
    env = {**os.environ, "TABPFN_TOKEN": ""}
    args = ["--model", "tabpfn3p5", "--dataset_name", "MUL_IMAGE_PETFINDER", "--fold", "0", "--output_dir", tmp_path]
    done = subprocess.run([sys.executable, "benchmark.py", *map(str, args)], cwd=REPO_DIR, env=env, capture_output=True,
                          text=True)
    assert done.returncode == 2 and "TABPFN_TOKEN" in done.stderr and not list(tmp_path.iterdir())


def test_a_rejected_tabpfn_token_stops_the_run(monkeypatch):
    browser_auth = pytest.importorskip("tabpfn.browser_auth")
    from multabench.baselines.tabpfn import TabPFN3p5, TabPFNv2
    monkeypatch.setenv("TABPFN_TOKEN", "invalid")
    monkeypatch.setattr(browser_auth, "verify_token", lambda token, api_url: False)
    assert "rejects TABPFN_TOKEN" in TabPFN3p5.env_error() and TabPFNv2.env_error() is None
    monkeypatch.setattr(browser_auth, "verify_token", lambda token, api_url: None)
    assert TabPFN3p5.env_error() is None
