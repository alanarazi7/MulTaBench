import os

from dotenv import load_dotenv

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN or ""
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
GPU = os.getenv("GPU")
IMG_DEBUG = os.getenv("IMG_DEBUG") is not None

DEVICE = None
if GPU is not None:
    DEVICE = f"cuda:{GPU}"
