"""Project discovery/parsing + ``Environments.create`` from a local project
(no live workspace)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.environments import service as W
from yggdrasil.databricks.environments.service import Environments


def _write_pyproject(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pyproject.toml"
    path.write_text(body, encoding="utf-8")
    return path


# ── discovery + parsing ────────────────────────────────────────────────────


def test_find_pyproject_walks_up(tmp_path):
    root = tmp_path / "proj"
    _write_pyproject(root, "[project]\nname='p'\n")
    nested = root / "src" / "pkg"
    nested.mkdir(parents=True)
    assert W.find_pyproject(nested) == root / "pyproject.toml"
    assert W.find_pyproject(root / "pyproject.toml") == root / "pyproject.toml"


def test_find_pyproject_none_when_absent(tmp_path):
    # A real path with no project on the way up → None (a PyPI name, not a path).
    assert W.find_pyproject(tmp_path) is None
    # A non-existent path (e.g. a bare PyPI name) is not a local path → None.
    assert W.find_pyproject("ygg") is None


def test_read_pyproject_extracts_name_version_deps_and_extras(tmp_path):
    path = _write_pyproject(
        tmp_path,
        '[project]\n'
        'name = "my-proj"\n'
        'version = "1.2.3"\n'
        'requires-python = ">=3.10"\n'
        'dependencies = ["polars>=1.0", "httpx"]\n\n'
        '[project.optional-dependencies]\n'
        'databricks = ["databricks-sdk"]\n',
    )
    meta = W.read_pyproject(path)
    assert meta["name"] == "my-proj"
    assert meta["version"] == "1.2.3"
    assert meta["dependencies"] == ["polars>=1.0", "httpx"]
    assert meta["optional_dependencies"]["databricks"] == ["databricks-sdk"]
    assert meta["requires_python"] == ">=3.10"
    assert meta["dir"] == tmp_path.resolve()


def test_read_pyproject_requires_a_name(tmp_path):
    path = _write_pyproject(tmp_path, "[project]\nversion = '0.1'\n")
    with pytest.raises(ValueError):
        W.read_pyproject(path)


# ── Environments.create — local project → zero-PyPI base environment ─────────


def test_create_builds_env_from_local_project(tmp_path):
    _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\nversion = "0.1.0"\ndependencies = ["polars"]\n',
    )
    client = MagicMock()
    svc = Environments(client=client)
    with patch.object(W, "fetch_wheels",
                      return_value=[Path("/tmp/demo-0.1.0-py3-none-any.whl"),
                                    Path("/tmp/polars-1.0-cp312.whl")]) as fetch, \
         patch.object(W, "registry_upload", side_effect=lambda c, w, **k: f"/ws/pypi/{Path(w).name}"), \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml") as named, \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req") as req:
        env = svc.create(str(tmp_path))

    # Discovered the local project + fetched its whole closure (zero-PyPI).
    assert fetch.call_args.kwargs["deps"] is True
    assert env.project == "demo"
    assert str(env.version) == "0.1.0"
    assert env.name == f"demo-0.1.0-{W.environment_key_for(None)}"
    assert env.serverless == "/env/demo.yml"
    assert env.cluster == "/env/demo.req"
    assert env.dependencies == [
        "/ws/pypi/demo-0.1.0-py3-none-any.whl", "/ws/pypi/polars-1.0-cp312.whl",
    ]
    assert named.call_args.kwargs["dependencies"] == env.dependencies
    assert req.call_args.kwargs["dependencies"] == env.dependencies


def test_create_from_pypi_name_derives_version_from_wheel(tmp_path):
    client = MagicMock()
    svc = Environments(client=client)
    with patch.object(W, "fetch_wheels",
                      return_value=[Path("/tmp/polars-1.2.3-cp312.whl")]), \
         patch.object(W, "registry_upload", side_effect=lambda c, w, **k: f"/ws/pypi/{Path(w).name}"), \
         patch.object(W, "ensure_named_environment", return_value="/env/polars.yml"), \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/polars.req"):
        env = svc.create("polars")
    # No pyproject → a PyPI name; version is read back off the fetched wheel.
    assert env.project == "polars"
    assert str(env.version) == "1.2.3"
