import subprocess
from typing import Dict

import wandb
from wandb.errors import CommError

from multabench.constants import WANDB_API_KEY, WANDB_ENTITY

def get_current_commit_hash() -> str:
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit_hash[:7]
    except subprocess.CalledProcessError:
        print(f"🆔 Could not get the current commit hash.")
        return ""


def wandb_run(exp_name: str, project: str) -> None:
    if WANDB_API_KEY is None:
        raise EnvironmentError("WANDB_API_KEY not found in .env file — refusing to run without WandB logging.")
    if WANDB_ENTITY is None:
        raise EnvironmentError("WANDB_ENTITY not found in .env file — refusing to run without WandB logging.")
    wandb.login(key=WANDB_API_KEY)
    try:
        wandb.init(entity=WANDB_ENTITY, project=project, reinit=True, name=exp_name)
    except CommError as e:
        print(f"⚠️ WandB couldn't connect to entity `{WANDB_ENTITY}`!")
        raise e


def wandb_finish(d_summary: Dict):
    wandb.log(d_summary)
    wandb.summary.update(d_summary)
    wandb.finish()
    print(f"Summary logged to WandB: {d_summary}")
