import subprocess

import pytest

from multabench.utils.logging import MASTER, assert_official_commit


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init")
    (tmp_path / "a.py").write_text("a = 1\n")
    git(tmp_path, "add", "a.py")
    git(tmp_path, "commit", "-m", "a")
    git(tmp_path, "update-ref", f"refs/remotes/{MASTER}", "HEAD")
    return tmp_path


def test_a_clean_commit_on_master_is_official(repo):
    (repo / "untracked.txt").write_text("jobs\n")
    assert_official_commit(cwd=repo)


def test_uncommitted_changes_are_not_official(repo):
    (repo / "a.py").write_text("a = 2\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        assert_official_commit(cwd=repo)


def test_a_commit_off_master_is_not_official(repo):
    git(repo, "commit", "--allow-empty", "-m", "branch")
    with pytest.raises(RuntimeError, match="not on"):
        assert_official_commit(cwd=repo)
