"""Cluster-mount fast path tests for :class:`WorkspacePath`.

Inside a Databricks runtime, ``/Workspace/...`` is exposed as a FUSE
mount. :class:`WorkspacePath` transparently detects the mount and
short-circuits stat/read/ls/mkdir/remove onto stdlib filesystem
calls so reads/writes don't pay a Workspace REST API round trip per
operation. Notebook upload semantics stay subtle, so the upload path
keeps going through ``workspace.upload`` even on cluster.

Mount detection lives behind
:func:`yggdrasil.databricks.fs.workspace_path._workspace_mount_available`
which ANDs ``DatabricksClient.is_in_databricks_environment()`` with
``os.path.isdir('/Workspace')``. Tests monkey-patch the helper to
emulate "on cluster" without actually running inside DBR.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.fs import workspace_path as wp_module
from yggdrasil.databricks.fs.workspace_path import WorkspacePath
from yggdrasil.databricks.workspaces.service import Workspaces
from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.io.io_stats import IOKind


@pytest.fixture(autouse=True)
def reset_workspace_path_caches():
    WorkspacePath._INSTANCES.clear()
    wp_module._reset_workspace_mount_probe()
    yield
    WorkspacePath._INSTANCES.clear()
    wp_module._reset_workspace_mount_probe()


@pytest.fixture
def fake_workspace_root(tmp_path, monkeypatch):
    """Stage a fake ``/Workspace`` tree under ``tmp_path`` and
    redirect :attr:`WorkspacePath.api_path` to it. ``_resolve_me``
    inside ``full_path`` is bypassed too by keeping the path free of
    ``<me>`` placeholders."""
    root = tmp_path / "Workspace" / "Users" / "alice"
    root.mkdir(parents=True)

    monkeypatch.setattr(
        wp_module, "_workspace_mount_available", lambda: True,
    )

    original = WorkspacePath.api_path

    def fake_api_path(self):
        full = original.fget(self)
        return str(tmp_path / full.lstrip("/"))

    monkeypatch.setattr(WorkspacePath, "api_path", property(fake_api_path))
    return root


@pytest.fixture
def service():
    client = MagicMock()
    svc = MagicMock(spec=Workspaces)
    svc.client = client
    return svc


# ===========================================================================
# Probe semantics
# ===========================================================================


class TestWorkspaceMountProbe:

    def test_probe_returns_false_when_not_in_runtime(self, monkeypatch):
        from yggdrasil.databricks.client import DatabricksClient
        monkeypatch.setattr(
            DatabricksClient, "is_in_databricks_environment",
            staticmethod(lambda: False),
        )
        assert wp_module._workspace_mount_available() is False

    def test_probe_returns_false_when_workspace_dir_missing(
        self, monkeypatch,
    ):
        from yggdrasil.databricks.client import DatabricksClient
        monkeypatch.setattr(
            DatabricksClient, "is_in_databricks_environment",
            staticmethod(lambda: True),
        )
        import os
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        wp_module._reset_workspace_mount_probe()
        assert wp_module._workspace_mount_available() is False

    def test_probe_result_is_cached(self, monkeypatch):
        from yggdrasil.databricks.client import DatabricksClient
        calls = {"n": 0}

        def is_runtime():
            calls["n"] += 1
            return False

        monkeypatch.setattr(
            DatabricksClient, "is_in_databricks_environment",
            staticmethod(is_runtime),
        )
        wp_module._workspace_mount_available()
        wp_module._workspace_mount_available()
        wp_module._workspace_mount_available()
        assert calls["n"] == 1


# ===========================================================================
# _stat_uncached
# ===========================================================================


class TestStatFastPath:

    def test_stat_file_uses_mount(
        self, fake_workspace_root, service,
    ):
        (fake_workspace_root / "config.txt").write_bytes(b"hi" * 10)
        p = WorkspacePath(
            "/Workspace/Users/alice/config.txt", service=service,
        )
        stats = p._stat_uncached()
        assert stats.kind is IOKind.FILE
        assert stats.size == 20
        service.client.workspace_client.assert_not_called()

    def test_stat_directory_uses_mount(
        self, fake_workspace_root, service,
    ):
        (fake_workspace_root / "repo").mkdir()
        p = WorkspacePath(
            "/Workspace/Users/alice/repo", service=service,
        )
        stats = p._stat_uncached()
        assert stats.kind is IOKind.DIRECTORY
        service.client.workspace_client.assert_not_called()

    def test_stat_missing_returns_missing(
        self, fake_workspace_root, service,
    ):
        p = WorkspacePath(
            "/Workspace/Users/alice/nope.bin", service=service,
        )
        stats = p._stat_uncached()
        assert stats.kind is IOKind.MISSING
        service.client.workspace_client.assert_not_called()


# ===========================================================================
# _read_mv
# ===========================================================================


class TestReadMvFastPath:

    def test_whole_file_read(self, fake_workspace_root, service):
        (fake_workspace_root / "data.txt").write_bytes(b"workspace-bytes")
        p = WorkspacePath(
            "/Workspace/Users/alice/data.txt", service=service,
        )
        mv = p._read_mv(-1, 0)
        assert bytes(mv) == b"workspace-bytes"
        service.client.workspace_client.assert_not_called()

    def test_positional_read(self, fake_workspace_root, service):
        (fake_workspace_root / "data.txt").write_bytes(b"abcdefghij")
        p = WorkspacePath(
            "/Workspace/Users/alice/data.txt", service=service,
        )
        mv = p._read_mv(3, 4)
        assert bytes(mv) == b"efg"

    def test_read_missing_raises_filenotfound(
        self, fake_workspace_root, service,
    ):
        p = WorkspacePath(
            "/Workspace/Users/alice/missing", service=service,
        )
        with pytest.raises(FileNotFoundError):
            p._read_mv(-1, 0)


# ===========================================================================
# _ls
# ===========================================================================


class TestLsFastPath:

    def test_ls_returns_children_with_seeded_stat_cache(
        self, fake_workspace_root, service,
    ):
        (fake_workspace_root / "a.py").write_bytes(b"print(1)")
        (fake_workspace_root / "b.txt").write_bytes(b"hello")
        (fake_workspace_root / "sub").mkdir()
        p = WorkspacePath(
            "/Workspace/Users/alice", service=service,
        )
        children = list(p._ls())
        names = sorted(c.name for c in children)
        assert names == ["a.py", "b.txt", "sub"]
        by_name = {c.name: c for c in children}
        assert by_name["a.py"]._stat_cached.kind is IOKind.FILE
        assert by_name["a.py"]._stat_cached.size == 8
        assert by_name["sub"]._stat_cached.kind is IOKind.DIRECTORY
        service.client.workspace_client.assert_not_called()

    def test_ls_recursive(self, fake_workspace_root, service):
        (fake_workspace_root / "sub").mkdir()
        (fake_workspace_root / "sub" / "deep.py").write_bytes(b"x = 1")
        p = WorkspacePath(
            "/Workspace/Users/alice", service=service,
        )
        names = sorted(c.name for c in p._ls(recursive=True))
        assert names == ["deep.py", "sub"]


# ===========================================================================
# _mkdir / _remove_file / _remove_dir
# ===========================================================================


class TestMkdirFastPath:

    def test_mkdir(self, fake_workspace_root, service):
        p = WorkspacePath(
            "/Workspace/Users/alice/newdir", service=service,
        )
        p._mkdir(parents=False, exist_ok=False)
        assert (fake_workspace_root / "newdir").is_dir()
        service.client.workspace_client.assert_not_called()

    def test_mkdir_parents(self, fake_workspace_root, service):
        p = WorkspacePath(
            "/Workspace/Users/alice/a/b/c", service=service,
        )
        p._mkdir(parents=True, exist_ok=False)
        assert (fake_workspace_root / "a" / "b" / "c").is_dir()

    def test_mkdir_exist_ok_swallows(self, fake_workspace_root, service):
        (fake_workspace_root / "existing").mkdir()
        p = WorkspacePath(
            "/Workspace/Users/alice/existing", service=service,
        )
        p._mkdir(parents=False, exist_ok=True)  # should not raise

    def test_mkdir_exist_ok_false_raises(
        self, fake_workspace_root, service,
    ):
        (fake_workspace_root / "existing").mkdir()
        p = WorkspacePath(
            "/Workspace/Users/alice/existing", service=service,
        )
        with pytest.raises(FileExistsError):
            p._mkdir(parents=False, exist_ok=False)


class TestRemoveFastPath:

    def test_remove_file(self, fake_workspace_root, service):
        target = fake_workspace_root / "f.py"
        target.write_bytes(b"x")
        p = WorkspacePath(
            "/Workspace/Users/alice/f.py", service=service,
        )
        p._remove_file(missing_ok=False, wait=WaitingConfig.from_(True))
        assert not target.exists()
        service.client.workspace_client.assert_not_called()

    def test_remove_file_missing_ok(self, fake_workspace_root, service):
        p = WorkspacePath(
            "/Workspace/Users/alice/ghost", service=service,
        )
        p._remove_file(missing_ok=True, wait=WaitingConfig.from_(True))

    def test_remove_file_missing_raises(
        self, fake_workspace_root, service,
    ):
        p = WorkspacePath(
            "/Workspace/Users/alice/ghost", service=service,
        )
        with pytest.raises(FileNotFoundError):
            p._remove_file(
                missing_ok=False, wait=WaitingConfig.from_(True),
            )

    def test_remove_empty_dir(self, fake_workspace_root, service):
        (fake_workspace_root / "empty").mkdir()
        p = WorkspacePath(
            "/Workspace/Users/alice/empty", service=service,
        )
        p._remove_dir(
            recursive=False,
            missing_ok=False,
            wait=WaitingConfig.from_(True),
        )
        assert not (fake_workspace_root / "empty").exists()
        service.client.workspace_client.assert_not_called()

    def test_remove_dir_recursive(self, fake_workspace_root, service):
        sub = fake_workspace_root / "tree"
        sub.mkdir()
        (sub / "a").write_bytes(b"A")
        (sub / "inner").mkdir()
        (sub / "inner" / "b").write_bytes(b"B")
        p = WorkspacePath(
            "/Workspace/Users/alice/tree", service=service,
        )
        p._remove_dir(
            recursive=True,
            missing_ok=False,
            wait=WaitingConfig.from_(True),
        )
        assert not sub.exists()


# ===========================================================================
# Off-cluster fallback
# ===========================================================================


class TestWorkspaceApiStillUsedOffCluster:
    """When the mount probe returns False, every operation still goes
    through ``workspace.workspace.*`` — the fast path is opt-in by
    mount detection only."""

    def test_stat_falls_back(self, monkeypatch, service):
        monkeypatch.setattr(
            wp_module, "_workspace_mount_available", lambda: False,
        )
        info = MagicMock()
        info.object_type = "FILE"
        info.size = 42
        info.modified_at = 0
        service.client.workspace_client.return_value.workspace.get_status \
            .return_value = info
        p = WorkspacePath(
            "/Workspace/Users/alice/file", service=service,
        )
        stats = p._stat_uncached()
        assert stats.size == 42
        service.client.workspace_client.return_value.workspace \
            .get_status.assert_called_once()


# ===========================================================================
# OSError fallback inside the fast path
# ===========================================================================


class TestOSErrorFallback:

    def test_read_oserror_falls_back_to_api(
        self, fake_workspace_root, monkeypatch, service,
    ):
        # ``open`` raising a non-FileNotFoundError OSError must route
        # the request through ``workspace.download`` instead of
        # crashing — covers permission errors and transient mount
        # layer glitches.
        import builtins
        real_open = builtins.open

        def bad_open(path, *a, **kw):
            if "/Workspace/Users/alice/explode" in str(path):
                raise PermissionError("simulated")
            return real_open(path, *a, **kw)

        monkeypatch.setattr(builtins, "open", bad_open)
        response = MagicMock()
        response.contents = MagicMock()
        response.contents.read.return_value = b"recovered"
        service.client.workspace_client.return_value.workspace.download \
            .return_value = response
        p = WorkspacePath(
            "/Workspace/Users/alice/explode", service=service,
        )
        mv = p._read_mv(-1, 0)
        assert bytes(mv) == b"recovered"
