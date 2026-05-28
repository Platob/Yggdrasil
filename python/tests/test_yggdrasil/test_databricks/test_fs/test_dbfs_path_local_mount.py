"""Cluster-mount fast path tests for :class:`DBFSPath`.

The classic ``/dbfs`` FUSE mount on DBR clusters lets us read / write
/ stat / list / mkdir / remove DBFS objects via plain stdlib syscalls,
bypassing the 1 MiB chunked + base64-encoded ``dbfs.read`` / SDK
``dbfs.open(write=True)`` REST API. Mount detection ANDs
``DatabricksClient.is_in_databricks_environment()`` with
``os.path.isdir('/dbfs')``.

Tests monkey-patch the probe to emulate "on cluster" without
actually running inside DBR. Note that :attr:`DBFSPath.api_path`
strips the ``/dbfs/`` prefix for the SDK; the mount fast path uses
:meth:`DBFSPath.full_path` which keeps the prefix.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.fs import dbfs_path as dp_module
from yggdrasil.databricks.fs.dbfs_path import DBFSPath
from yggdrasil.databricks.fs.service import DBFSService
from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.io.io_stats import IOKind


@pytest.fixture(autouse=True)
def reset_dbfs_path_caches():
    DBFSPath._INSTANCES.clear()
    dp_module._reset_dbfs_mount_probe()
    yield
    DBFSPath._INSTANCES.clear()
    dp_module._reset_dbfs_mount_probe()


@pytest.fixture
def fake_dbfs_root(tmp_path, monkeypatch):
    """Stage a fake ``/dbfs`` tree under ``tmp_path`` and redirect
    :meth:`DBFSPath.full_path` (the kernel-mount path) to it."""
    root = tmp_path / "dbfs" / "tmp"
    root.mkdir(parents=True)

    monkeypatch.setattr(
        dp_module, "_dbfs_mount_available", lambda: True,
    )

    original = DBFSPath.full_path
    tmp_prefix = str(tmp_path)

    def fake_full_path(self):
        full = original(self)
        # ``_ls`` recursively builds child paths from this method's
        # output — make the patch idempotent so re-translation on
        # the recursive call doesn't double-prefix the tmp dir.
        if full.startswith(tmp_prefix):
            return full
        return str(tmp_path / full.lstrip("/"))

    monkeypatch.setattr(DBFSPath, "full_path", fake_full_path)
    return root


@pytest.fixture
def service():
    client = MagicMock()
    svc = MagicMock(spec=DBFSService)
    svc.client = client
    return svc


# ===========================================================================
# Probe
# ===========================================================================


class TestDbfsMountProbe:

    def test_probe_false_off_runtime(self, monkeypatch):
        from yggdrasil.databricks.client import DatabricksClient
        monkeypatch.setattr(
            DatabricksClient, "is_in_databricks_environment",
            staticmethod(lambda: False),
        )
        assert dp_module._dbfs_mount_available() is False

    def test_probe_false_when_dbfs_dir_missing(self, monkeypatch):
        from yggdrasil.databricks.client import DatabricksClient
        monkeypatch.setattr(
            DatabricksClient, "is_in_databricks_environment",
            staticmethod(lambda: True),
        )
        import os
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        dp_module._reset_dbfs_mount_probe()
        assert dp_module._dbfs_mount_available() is False

    def test_probe_cached(self, monkeypatch):
        from yggdrasil.databricks.client import DatabricksClient
        calls = {"n": 0}

        def is_runtime():
            calls["n"] += 1
            return False

        monkeypatch.setattr(
            DatabricksClient, "is_in_databricks_environment",
            staticmethod(is_runtime),
        )
        dp_module._dbfs_mount_available()
        dp_module._dbfs_mount_available()
        assert calls["n"] == 1


# ===========================================================================
# _stat_uncached
# ===========================================================================


class TestStatFastPath:

    def test_stat_file(self, fake_dbfs_root, service):
        (fake_dbfs_root / "data").write_bytes(b"x" * 9)
        p = DBFSPath("/dbfs/tmp/data", service=service)
        stats = p._stat_uncached()
        assert stats.kind is IOKind.FILE
        assert stats.size == 9
        service.client.workspace_client.assert_not_called()

    def test_stat_directory(self, fake_dbfs_root, service):
        (fake_dbfs_root / "subdir").mkdir()
        p = DBFSPath("/dbfs/tmp/subdir", service=service)
        stats = p._stat_uncached()
        assert stats.kind is IOKind.DIRECTORY

    def test_stat_missing(self, fake_dbfs_root, service):
        p = DBFSPath("/dbfs/tmp/nope", service=service)
        stats = p._stat_uncached()
        assert stats.kind is IOKind.MISSING


# ===========================================================================
# _read_mv (the biggest win — DBFS REST chunks at 1 MiB w/ base64)
# ===========================================================================


class TestReadMvFastPath:

    def test_whole_file_read(self, fake_dbfs_root, service):
        payload = b"X" * (5 * 1024 * 1024)  # 5 MiB > one DBFS chunk
        (fake_dbfs_root / "big").write_bytes(payload)
        p = DBFSPath("/dbfs/tmp/big", service=service)
        mv = p._read_mv(-1, 0)
        assert bytes(mv) == payload
        service.client.workspace_client.assert_not_called()

    def test_positional_read(self, fake_dbfs_root, service):
        (fake_dbfs_root / "f").write_bytes(b"abcdefghij")
        p = DBFSPath("/dbfs/tmp/f", service=service)
        mv = p._read_mv(3, 4)
        assert bytes(mv) == b"efg"

    def test_read_missing_raises(self, fake_dbfs_root, service):
        p = DBFSPath("/dbfs/tmp/ghost", service=service)
        with pytest.raises(FileNotFoundError):
            p._read_mv(-1, 0)


# ===========================================================================
# _ls
# ===========================================================================


class TestLsFastPath:

    def test_ls_seeds_child_stat_caches(self, fake_dbfs_root, service):
        (fake_dbfs_root / "a").write_bytes(b"AA")
        (fake_dbfs_root / "b").write_bytes(b"BBBB")
        (fake_dbfs_root / "sub").mkdir()
        p = DBFSPath("/dbfs/tmp", service=service)
        children = list(p._ls())
        names = sorted(c.name for c in children)
        assert names == ["a", "b", "sub"]
        by_name = {c.name: c for c in children}
        assert by_name["a"]._stat_cached.kind is IOKind.FILE
        assert by_name["a"]._stat_cached.size == 2
        assert by_name["sub"]._stat_cached.kind is IOKind.DIRECTORY
        service.client.workspace_client.assert_not_called()

    def test_ls_recursive(self, fake_dbfs_root, service):
        (fake_dbfs_root / "sub").mkdir()
        (fake_dbfs_root / "sub" / "leaf").write_bytes(b"L")
        p = DBFSPath("/dbfs/tmp", service=service)
        names = sorted(c.name for c in p._ls(recursive=True))
        assert names == ["leaf", "sub"]


# ===========================================================================
# Mutations
# ===========================================================================


class TestMkdirFastPath:

    def test_mkdir(self, fake_dbfs_root, service):
        p = DBFSPath("/dbfs/tmp/newdir", service=service)
        p._mkdir(parents=False, exist_ok=False)
        assert (fake_dbfs_root / "newdir").is_dir()

    def test_mkdir_parents(self, fake_dbfs_root, service):
        p = DBFSPath("/dbfs/tmp/deep/sub", service=service)
        p._mkdir(parents=True, exist_ok=False)
        assert (fake_dbfs_root / "deep" / "sub").is_dir()

    def test_mkdir_exist_ok_swallows(self, fake_dbfs_root, service):
        (fake_dbfs_root / "exists").mkdir()
        p = DBFSPath("/dbfs/tmp/exists", service=service)
        p._mkdir(parents=False, exist_ok=True)  # no raise

    def test_mkdir_exist_ok_false_raises(self, fake_dbfs_root, service):
        (fake_dbfs_root / "exists").mkdir()
        p = DBFSPath("/dbfs/tmp/exists", service=service)
        with pytest.raises(FileExistsError):
            p._mkdir(parents=False, exist_ok=False)


class TestRemoveFastPath:

    def test_remove_file(self, fake_dbfs_root, service):
        (fake_dbfs_root / "f").write_bytes(b"x")
        p = DBFSPath("/dbfs/tmp/f", service=service)
        p._remove_file(missing_ok=False, wait=WaitingConfig.from_(True))
        assert not (fake_dbfs_root / "f").exists()

    def test_remove_file_missing_ok(self, fake_dbfs_root, service):
        p = DBFSPath("/dbfs/tmp/ghost", service=service)
        p._remove_file(missing_ok=True, wait=WaitingConfig.from_(True))

    def test_remove_file_missing_raises(self, fake_dbfs_root, service):
        p = DBFSPath("/dbfs/tmp/ghost", service=service)
        with pytest.raises(FileNotFoundError):
            p._remove_file(
                missing_ok=False, wait=WaitingConfig.from_(True),
            )

    def test_remove_empty_dir(self, fake_dbfs_root, service):
        (fake_dbfs_root / "empty").mkdir()
        p = DBFSPath("/dbfs/tmp/empty", service=service)
        p._remove_dir(
            recursive=False,
            missing_ok=False,
            wait=WaitingConfig.from_(True),
        )
        assert not (fake_dbfs_root / "empty").exists()

    def test_remove_dir_recursive(self, fake_dbfs_root, service):
        sub = fake_dbfs_root / "tree"
        sub.mkdir()
        (sub / "a").write_bytes(b"A")
        (sub / "inner").mkdir()
        (sub / "inner" / "b").write_bytes(b"B")
        p = DBFSPath("/dbfs/tmp/tree", service=service)
        p._remove_dir(
            recursive=True,
            missing_ok=False,
            wait=WaitingConfig.from_(True),
        )
        assert not sub.exists()


# ===========================================================================
# Upload
# ===========================================================================


class TestUploadFastPath:

    def test_upload_bytes(self, fake_dbfs_root, service):
        p = DBFSPath("/dbfs/tmp/new", service=service)
        n = p._upload(b"hello-dbfs")
        assert n == 10
        assert (fake_dbfs_root / "new").read_bytes() == b"hello-dbfs"
        service.client.workspace_client.assert_not_called()

    def test_upload_creates_parents(self, fake_dbfs_root, service):
        p = DBFSPath("/dbfs/tmp/a/b/c", service=service)
        p._upload(b"deep")
        assert (
            fake_dbfs_root / "a" / "b" / "c"
        ).read_bytes() == b"deep"

    def test_upload_streaming(self, fake_dbfs_root, service):
        import io
        p = DBFSPath("/dbfs/tmp/streamed", service=service)
        n = p._upload(io.BytesIO(b"streamed-bytes"))
        assert n == 14
        assert (
            fake_dbfs_root / "streamed"
        ).read_bytes() == b"streamed-bytes"


# ===========================================================================
# Off-cluster fallback
# ===========================================================================


class TestDbfsApiStillUsedOffCluster:

    def test_stat_falls_back(self, monkeypatch, service):
        monkeypatch.setattr(
            dp_module, "_dbfs_mount_available", lambda: False,
        )
        info = MagicMock()
        info.is_dir = False
        info.file_size = 99
        info.modification_time = 0
        service.client.workspace_client.return_value.dbfs.get_status \
            .return_value = info
        p = DBFSPath("/dbfs/tmp/f", service=service)
        stats = p._stat_uncached()
        assert stats.size == 99
        service.client.workspace_client.return_value.dbfs.get_status \
            .assert_called_once()
