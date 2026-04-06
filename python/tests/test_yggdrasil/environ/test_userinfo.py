# tests/test_userinfo.py
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture()
def userinfo(monkeypatch):
    modname = "yggdrasil.environ.userinfo"  # change if your module name differs
    m = importlib.import_module(modname)
    importlib.reload(m)
    m._clear_cache()
    monkeypatch.setattr(m.socket, "gethostname", lambda: "hostA")
    return m


def test_cache(userinfo, monkeypatch):
    calls = {"key": 0, "email": 0, "cwd": 0, "git": 0, "proj": 0, "url": 0}

    monkeypatch.setattr(userinfo, "_get_key", lambda: calls.__setitem__("key", calls["key"] + 1) or "k")
    monkeypatch.setattr(userinfo, "_get_upn_email", lambda: calls.__setitem__("email", calls["email"] + 1) or "e@x.com")
    monkeypatch.setattr(userinfo, "_guess_email_from_env", lambda: None)
    monkeypatch.setattr(userinfo, "_safe_getcwd", lambda: calls.__setitem__("cwd", calls["cwd"] + 1) or "/tmp/x")
    monkeypatch.setattr(userinfo, "_infer_project", lambda cwd: calls.__setitem__("proj", calls["proj"] + 1) or (None, None))
    monkeypatch.setattr(userinfo, "_git_info", lambda cwd: calls.__setitem__("git", calls["git"] + 1) or None)
    monkeypatch.setattr(userinfo, "_git_url_from_info", lambda git: None)
    monkeypatch.setattr(
        userinfo,
        "_current_compute_url",
        lambda *, hostname, cwd: calls.__setitem__("url", calls["url"] + 1) or None,
    )

    a = userinfo.get_user_info()
    b = userinfo.get_user_info()
    c = userinfo.get_user_info(refresh=True)

    assert a is b
    assert a is not c
    assert calls["key"] == 2
    assert calls["email"] == 2
    assert calls["cwd"] == 2
    assert calls["proj"] == 2
    assert calls["git"] == 2
    assert calls["url"] == 2


def test_normalize_abs_path_for_url(userinfo):
    assert userinfo.normalize_abs_path_for_url("/a//b///c") == "/a/b/c"
    assert userinfo.normalize_abs_path_for_url(r"C:\Users\Nika\proj") == "/C:/Users/Nika/proj"
    assert userinfo.normalize_abs_path_for_url(r"\\server\share\dir\file") == "//server/share/dir/file"


def test_databricks_url_prefers_job_run(userinfo, monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    monkeypatch.setattr(
        userinfo,
        "_ctx_tags",
        lambda: {
            "browserHostName": "adb-1234.56.azuredatabricks.net",
            "orgId": "9999999999999999",
            "jobId": "12",
            "jobRunId": "345",
            "notebookId": "777",
            "notebookPath": "/Users/nika/a",
        },
    )
    u = userinfo._databricks_current_url(kind="auto")
    assert u is not None
    assert u.to_string() == "https://adb-1234.56.azuredatabricks.net/?o=9999999999999999#job/12/run/345"


def test_current_compute_url_falls_back_to_local_cwd(userinfo, monkeypatch, tmp_path: Path):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.setattr(userinfo, "_databricks_current_url", lambda kind="auto": None)
    u = userinfo._current_compute_url(hostname="hostA", cwd=str(tmp_path))
    assert u.to_string() == f"local://hosta/{tmp_path.as_posix()}"


def test_git_inference_and_git_url(userinfo, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    gitdir = repo / ".git"
    (gitdir / "refs" / "heads").mkdir(parents=True)

    (gitdir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (gitdir / "refs" / "heads" / "main").write_text("0123456789abcdef0123456789abcdef01234567\n", encoding="utf-8")
    (gitdir / "config").write_text('[remote "origin"]\n  url = git@github.com:acme/my-repo.git\n', encoding="utf-8")

    info = userinfo._git_info(str(repo))
    assert info is not None
    assert info["git_remote"] == "git@github.com:acme/my-repo.git"
    assert info["git_branch"] == "main"
    assert info["git_sha"] == "0123456789ab"

    u = userinfo._git_url_from_info(info)
    assert u is not None
    assert u.to_string() == "https://github.com/acme/my-repo#0123456789ab"


def test_project_inference_pyproject_pep621(userinfo, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        """
[project]
name = "coolproj"
version = "1.2.3"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    name, ver = userinfo._infer_project(str(repo))
    assert name == "coolproj"
    assert ver == "1.2.3"


def test_project_inference_pyproject_poetry(userinfo, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        """
[tool.poetry]
name = "poetryproj"
version = "0.9.0"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    name, ver = userinfo._infer_project(str(repo))
    assert name == "poetryproj"
    assert ver == "0.9.0"


def test_project_inference_setup_py(userinfo, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "setup.py").write_text(
        """
from setuptools import setup

setup(
    name="setupproj",
    version="4.5.6",
)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    name, ver = userinfo._infer_project(str(repo))
    assert name == "setupproj"
    assert ver == "4.5.6"


def test_userinfo_integration_project_git_local_url(userinfo, monkeypatch, tmp_path: Path):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.setattr(userinfo, "_databricks_current_url", lambda kind="auto": None)
    monkeypatch.setattr(userinfo, "_get_key", lambda: "k")
    monkeypatch.setattr(userinfo, "_get_upn_email", lambda: None)
    monkeypatch.setattr(userinfo, "_guess_email_from_env", lambda: None)

    repo = tmp_path / "repo"
    repo.mkdir()

    # project
    (repo / "pyproject.toml").write_text(
        """
[project]
name = "coolproj"
version = "1.2.3"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    # git
    gitdir = repo / ".git"
    (gitdir / "refs" / "heads").mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (gitdir / "refs" / "heads" / "main").write_text("0123456789abcdef0123456789abcdef01234567\n", encoding="utf-8")
    (gitdir / "config").write_text('[remote "origin"]\n  url = git@github.com:acme/my-repo.git\n', encoding="utf-8")

    monkeypatch.setattr(userinfo, "_safe_getcwd", lambda: str(repo))

    ui = userinfo.get_user_info(refresh=True)
    assert ui.url.to_string() == f"local://hosta/{repo.as_posix()}"
    assert ui.git_url is not None
    assert ui.git_url.to_string() == "https://github.com/acme/my-repo#0123456789ab"
    assert ui.product == "coolproj"
    assert ui.product_version == "1.2.3"