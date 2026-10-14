import subprocess
import sys

# Not psutil, which joblib imports when it's installed, nor the bare `autogluon` namespace its .pth files register.
BENCH_ONLY = ["torch", "torchvision", "transformers", "peft", "PIL", "skrub", "autogluon.core", "dotenv"]


def test_the_core_imports_no_bench_package():
    code = ("import sys, multabench; from multabench import load_embeddings, load_split, Size; "
            f"print([m for m in {BENCH_ONLY} if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
