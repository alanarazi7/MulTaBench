import subprocess


def get_current_commit_hash() -> str:
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit_hash[:7]
    except subprocess.CalledProcessError:
        print(f"🆔 Could not get the current commit hash.")
        return ""

