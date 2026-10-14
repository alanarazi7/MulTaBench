import subprocess
from typing import Optional

MASTER = "origin/master"


def get_current_commit_hash() -> str:
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit_hash[:7]
    except subprocess.CalledProcessError:
        print(f"🆔 Could not get the current commit hash.")
        return ""


def assert_official_commit(cwd: Optional[str] = None):
    """Leaderboard runs need a clean commit on master, so their git column names code anyone can check out."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise RuntimeError("The checkout has uncommitted changes, which a leaderboard run can't have")
    if git("merge-base", "--is-ancestor", "HEAD", MASTER).returncode != 0:
        raise RuntimeError(f"HEAD is not on {MASTER}: launch leaderboard runs from a merged commit")
