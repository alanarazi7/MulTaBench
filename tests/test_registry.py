import os
import subprocess
import sys

REPO_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PACKAGES = ["autogluon.multimodal", "catboost", "causilo", "exaonetabular", "lightgbm", "pytabkit", "sap_rpt_oss",
                  "sdm", "tabdpt", "tabfm", "tabicl", "tabldm", "tabpfn", "tabpfn_extensions", "tabstar", "xgboost"]
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
