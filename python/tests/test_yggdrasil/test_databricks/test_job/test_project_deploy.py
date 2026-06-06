"""``ygg databricks deploy project`` building blocks — pyproject discovery,
parsing, and the project-named environment assembly (no live workspace)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.job import wheel as W


def _write_pyproject(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pyproject.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_find_pyproject_walks_up(tmp_path):
    root = tmp_path / "proj"
    _write_pyproject(root, "[project]\nname='p'\n")
    nested = root / "src" / "pkg"
    nested.mkdir(parents=True)
    # from a nested dir → the root pyproject
    assert W.find_pyproject(nested) == root / "pyproject.toml"
    # from the file itself → itself
    assert W.find_pyproject(root / "pyproject.toml") == root / "pyproject.toml"


def test_find_pyproject_raises_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        W.find_pyproject(tmp_path)


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


def test_ensure_project_environment_non_bundle_lists_wheel_then_index_deps(tmp_path):
    _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\nversion = "0.1.0"\n'
        'dependencies = ["polars"]\n\n'
        '[project.optional-dependencies]\nextra = ["httpx"]\n',
    )
    client = MagicMock()
    with patch.object(W, "build_project_wheel",
                      return_value=[Path("/tmp/demo-0.1.0-py3-none-any.whl")]), \
         patch.object(W, "registry_upload",
                      return_value="/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl"), \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml") as named, \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req") as req:
        info = W.ensure_project_environment(client, tmp_path, extras=("extra",))

    assert info["name"] == "demo"
    assert info["env_name"] == "demo-0-1-0"
    # wheel path first, then the project deps + requested extra (index requirements)
    assert info["dependencies"] == [
        "/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl", "polars", "httpx",
    ]
    assert info["serverless"] == "/env/demo.yml"
    assert info["cluster"] == "/env/demo.req"
    # both the serverless env and the cluster requirements got the same dep list
    assert named.call_args.kwargs["dependencies"] == info["dependencies"]
    assert req.call_args.kwargs["dependencies"] == info["dependencies"]


def test_ensure_project_environment_bundle_downloads_and_lists_wheels(tmp_path):
    _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\nversion = "0.1.0"\ndependencies = ["polars"]\n',
    )
    client = MagicMock()
    uploads = iter([
        "/env/demo-0-1-0/binaries/demo/demo-0.1.0-py3-none-any.whl",
        "/env/demo-0-1-0/binaries/polars/polars-1.0-cp312-...-.whl",
    ])
    with patch.object(W, "build_project_wheel",
                      return_value=[Path("/tmp/demo-0.1.0-py3-none-any.whl")]), \
         patch.object(W, "download_dependency_wheels",
                      return_value=[Path("/tmp/polars-1.0-cp312.whl")]) as dl, \
         patch.object(W, "registry_upload", side_effect=lambda *a, **k: next(uploads)), \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml"), \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req"):
        info = W.ensure_project_environment(client, tmp_path, bundle=True)

    dl.assert_called_once()                       # closure downloaded for zero-PyPI
    assert info["dependencies"] == [
        "/env/demo-0-1-0/binaries/demo/demo-0.1.0-py3-none-any.whl",
        "/env/demo-0-1-0/binaries/polars/polars-1.0-cp312-...-.whl",
    ]
