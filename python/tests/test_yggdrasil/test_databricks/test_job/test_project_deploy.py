"""``ygg databricks deploy project`` building blocks — pyproject discovery,
parsing, and the project-named environment assembly + deploy modes (no live
workspace)."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.environments import service as W
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


# ── ensure_project_environment — dependency composition ──────────────────────


def test_non_bundle_lists_wheel_then_index_deps(tmp_path):
    _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\nversion = "0.1.0"\n'
        'dependencies = ["polars"]\n\n'
        '[project.optional-dependencies]\nextra = ["httpx"]\n',
    )
    client = MagicMock()
    with patch.object(W, "deployed_wheels", return_value=[]), \
         patch.object(W, "build_project_wheel",
                      return_value=[Path("/tmp/demo-0.1.0-py3-none-any.whl")]), \
         patch.object(W, "registry_upload",
                      return_value="/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl"), \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml") as named, \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req") as req:
        info = W.ensure_project_environment(client, tmp_path, extras=("extra",))

    assert info["name"] == "demo"
    assert info["env_name"] == f"demo-0.1.0-{W.environment_key_for(None)}"
    assert info["env_dir"] == "/Workspace/Shared/environment/demo"
    assert info["mode"] == "AUTO"
    assert info["dependencies"] == [
        "/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl", "polars", "httpx",
    ]
    assert info["serverless"] == "/env/demo.yml"
    assert info["cluster"] == "/env/demo.req"
    assert named.call_args.kwargs["dependencies"] == info["dependencies"]
    assert req.call_args.kwargs["dependencies"] == info["dependencies"]


def test_bundle_downloads_and_lists_wheels(tmp_path):
    _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\nversion = "0.1.0"\ndependencies = ["polars"]\n',
    )
    client = MagicMock()
    uploads = iter([
        "/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl",
        "/Workspace/Shared/pypi/polars/polars-1.0-cp312.whl",
    ])
    with patch("yggdrasil.databricks.path.DatabricksPath", _fake_dbpath(exists=False)), \
         patch.object(W, "build_project_wheel",
                      return_value=[Path("/tmp/demo-0.1.0-py3-none-any.whl")]), \
         patch.object(W, "download_dependency_wheels",
                      return_value=[Path("/tmp/polars-1.0-cp312.whl")]) as dl, \
         patch.object(W, "registry_upload", side_effect=lambda *a, **k: next(uploads)), \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml"), \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req"):
        info = W.ensure_project_environment(client, tmp_path, bundle=True)

    dl.assert_called_once()
    assert info["dependencies"] == [
        "/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl",
        "/Workspace/Shared/pypi/polars/polars-1.0-cp312.whl",
    ]


# ── ensure_project_environment — deploy modes ────────────────────────────────


def _proj(tmp_path):
    _write_pyproject(
        tmp_path, '[project]\nname = "demo"\nversion = "0.1.0"\ndependencies = ["polars"]\n',
    )
    return MagicMock()


def test_auto_reuses_deployed_wheel_without_building(tmp_path):
    client = _proj(tmp_path)
    with patch.object(W, "deployed_wheels",
                      return_value=["/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl"]), \
         patch.object(W, "build_project_wheel") as build, \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml") as named, \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req") as req:
        info = W.ensure_project_environment(client, tmp_path, mode=Mode.AUTO)
    build.assert_not_called()                         # get-or-create → reused
    named.assert_called_once()                        # but env files overwritten
    req.assert_called_once()
    assert info["dependencies"][0].endswith("demo-0.1.0-py3-none-any.whl")


def test_overwrite_rebuilds_even_when_deployed(tmp_path):
    client = _proj(tmp_path)
    with patch.object(W, "deployed_wheels",
                      return_value=["/already/there.whl"]) as deployed, \
         patch.object(W, "build_project_wheel",
                      return_value=[Path("/tmp/demo-0.1.0-py3-none-any.whl")]) as build, \
         patch.object(W, "registry_upload",
                      return_value="/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl"), \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml"), \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req"):
        info = W.ensure_project_environment(client, tmp_path, mode=Mode.OVERWRITE)
    deployed.assert_not_called()                      # OVERWRITE never checks; always builds
    build.assert_called_once()
    assert info["mode"] == "OVERWRITE"


def test_append_writes_env_files_only_when_missing(tmp_path):
    client = _proj(tmp_path)
    # Serverless yml already present, requirements absent → only the missing one
    # is written.
    node_present = MagicMock(); node_present.exists.return_value = True
    node_absent = MagicMock(); node_absent.exists.return_value = False
    dbp = MagicMock()
    dbp.from_.side_effect = lambda dest, **k: (
        node_present if str(dest).endswith(".yml") else node_absent
    )
    with patch("yggdrasil.databricks.path.DatabricksPath", dbp), \
         patch.object(W, "deployed_wheels",
                      return_value=["/Workspace/Shared/pypi/demo/demo-0.1.0-py3-none-any.whl"]), \
         patch.object(W, "ensure_named_environment", return_value="/env/demo.yml") as named, \
         patch.object(W, "ensure_cluster_requirements", return_value="/env/demo.req") as req:
        info = W.ensure_project_environment(client, tmp_path, mode=Mode.APPEND)
    named.assert_not_called()                         # yml exists → left alone
    req.assert_called_once()                          # requirements missing → written
    assert info["serverless"].endswith(                    # reported existing path
        f"demo-0.1.0-{W.environment_key_for(None)}.yml"
    )
    assert info["mode"] == "APPEND"
