import os

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN or ""
GPU = os.getenv("GPU")

DEVICE = None
if GPU is not None:
    DEVICE = f"cuda:{GPU}"

SEED = 42
