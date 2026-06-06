"""``ygg databricks deploy project`` building blocks — pyproject discovery,
parsing, and the project-named environment assembly + deploy modes (no live
workspace)."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.job import wheel as W
from yggdrasil.enums.mode import Mode


def _write_pyproject(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pyproject.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _fake_dbpath(*, exists: bool = False, text: str = "") -> MagicMock:
    """A ``DatabricksPath`` stand-in whose ``.from_(...)`` yields a node with
    controllable ``exists()`` / ``read_text()``."""
    node = MagicMock()
    node.exists.return_value = exists
    node.read_text.return_value = text
    factory = MagicMock()
    factory.from_.return_value = node
    return factory


# ── discovery + parsing ────────────────────────────────────────────────────


def test_find_pyproject_walks_up(tmp_path):
    root = tmp_path / "proj"
    _write_pyproject(root, "[project]\nname='p'\n")
    nested = root / "src" / "pkg"
    nested.mkdir(parents=True)
    assert W.find_pyproject(nested) == root / "pyproject.toml"
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


# ── ensure_project_environment — closure to the shared registry ──────────────


def test_builds_project_wheel_then_dependency_wheels(tmp_path):
    """One unique way: the project wheel **and** its dependency closure are
    deployed as wheels into the shared registry, and the yml + requirements list
    those same paths (project wheel first)."""
    _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\nversion = "0.1.0"\n'
        'dependencies = ["polars"]\n\n'
        '[project.optional-dependencies]\nextra = ["httpx"]\n',
    )
    client = MagicMock()
    uploads = iter([
        "/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl",
        "/Workspace/Shared/pypi/polars/polars-1.0-cp312.whl",
        "/Workspace/Shared/pypi/httpx/httpx-0.27-py3-none-any.whl",
    ])
    with patch("yggdrasil.databricks.path.DatabricksPath", _fake_dbpath(exists=False)), \
         patch.object(W, "build_project_wheel",
                      return_value=[Path("/tmp/demo-0.1.0-py3-none-any.whl")]), \
         patch.object(W, "download_dependency_wheels",
                      return_value=[Path("/tmp/polars-1.0-cp312.whl"),
                                    Path("/tmp/httpx-0.27-py3-none-any.whl")]) as dl, \
         patch.object(W, "registry_upload", side_effect=lambda *a, **k: next(uploads)), \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml") as named, \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req") as req:
        info = W.ensure_project_environment(client, tmp_path, extras=("extra",))

    # extras flattened into the deps handed to the wheel downloader
    assert dl.call_args.args[0] == ["polars", "httpx"]
    assert info["name"] == "demo"
    assert info["env_name"] == "demo"          # version-free: named for the project alone
    assert info["mode"] == "AUTO"
    # every dependency is a wheel in the shared registry — project wheel first
    assert info["dependencies"] == [
        "/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl",
        "/Workspace/Shared/pypi/polars/polars-1.0-cp312.whl",
        "/Workspace/Shared/pypi/httpx/httpx-0.27-py3-none-any.whl",
    ]
    assert info["serverless"] == "/env/demo.yml"
    assert info["cluster"] == "/env/demo.req"
    # the yml and the requirements list the *same* shared-registry wheel paths
    assert named.call_args.kwargs["dependencies"] == info["dependencies"]
    assert req.call_args.kwargs["dependencies"] == info["dependencies"]


# ── ensure_project_environment — deploy modes ────────────────────────────────


def test_auto_reuses_cached_closure_without_building(tmp_path):
    _write_pyproject(
        tmp_path, '[project]\nname = "demo"\nversion = "0.1.0"\ndependencies = ["polars"]\n',
    )
    client = MagicMock()
    cached = (
        "/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl\n"
        "/Workspace/Shared/pypi/polars/polars-1.0-cp312.whl\n"
    )
    with patch("yggdrasil.databricks.path.DatabricksPath",
               _fake_dbpath(exists=True, text=cached)), \
         patch.object(W, "build_project_wheel") as build, \
         patch.object(W, "download_dependency_wheels") as dl, \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml") as named, \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req") as req:
        info = W.ensure_project_environment(client, tmp_path, mode=Mode.AUTO)
    build.assert_not_called()                         # cached closure reused
    dl.assert_not_called()
    named.assert_called_once()                        # but env files overwritten
    req.assert_called_once()
    assert info["dependencies"] == [
        "/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl",
        "/Workspace/Shared/pypi/polars/polars-1.0-cp312.whl",
    ]
    assert info["mode"] == "AUTO"


def test_overwrite_rebuilds_even_when_cached(tmp_path):
    _write_pyproject(
        tmp_path, '[project]\nname = "demo"\nversion = "0.1.0"\ndependencies = ["polars"]\n',
    )
    client = MagicMock()
    # A cached manifest is present, but OVERWRITE ignores it and rebuilds.
    with patch("yggdrasil.databricks.path.DatabricksPath",
               _fake_dbpath(exists=True, text="/old/cached.whl\n")), \
         patch.object(W, "build_project_wheel",
                      return_value=[Path("/tmp/demo-0.1.0-py3-none-any.whl")]) as build, \
         patch.object(W, "download_dependency_wheels", return_value=[]) as dl, \
         patch.object(W, "registry_upload",
                      return_value="/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl"), \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml"), \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req"):
        info = W.ensure_project_environment(client, tmp_path, mode=Mode.OVERWRITE)
    build.assert_called_once()                        # OVERWRITE always rebuilds
    dl.assert_called_once()
    assert info["mode"] == "OVERWRITE"
    assert info["dependencies"] == [
        "/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl",
    ]


def test_append_writes_env_files_only_when_missing(tmp_path):
    _write_pyproject(
        tmp_path, '[project]\nname = "demo"\nversion = "0.1.0"\ndependencies = ["polars"]\n',
    )
    client = MagicMock()
    # Closure cached (manifest present); serverless yml already present,
    # requirements absent → only the missing one is written.
    node_manifest = MagicMock()
    node_manifest.exists.return_value = True
    node_manifest.read_text.return_value = (
        "/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl\n"
    )
    node_present = MagicMock(); node_present.exists.return_value = True
    node_absent = MagicMock(); node_absent.exists.return_value = False

    def _from(dest, **k):
        d = str(dest)
        if d.endswith(".bundle"):
            return node_manifest
        if d.endswith(".whl") or d.endswith(".yml"):
            return node_present          # cached wheel + serverless yml exist
        return node_absent               # requirements.txt missing

    dbp = MagicMock(); dbp.from_.side_effect = _from
    with patch("yggdrasil.databricks.path.DatabricksPath", dbp), \
         patch.object(W, "build_project_wheel") as build, \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml") as named, \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req") as req:
        info = W.ensure_project_environment(client, tmp_path, mode=Mode.APPEND)
    build.assert_not_called()                         # closure reused
    named.assert_not_called()                         # yml exists → left alone
    req.assert_called_once()                          # requirements missing → written
    assert info["serverless"].endswith("demo.yml")   # reported existing path
    assert info["mode"] == "APPEND"
