"""Tests for Databricks path types: VolumePath, DBFSPath, WorkspacePath.

Mock tests verify construction, singleton caching, and minimum SDK
calls.  Integration tests (marked ``@pytest.mark.integration``) run
against a live workspace when ``DATABRICKS_HOST`` is set.
"""
from __future__ import annotations

import base64
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("databricks.sdk", reason="databricks-sdk not installed")

from yggdrasil.databricks.fs import DBFSPath, VolumePath, WorkspacePath  # noqa: E402
from yggdrasil.databricks.fs.service import DBFSService  # noqa: E402
from yggdrasil.databricks.volume.volumes import Volumes  # noqa: E402
from yggdrasil.databricks.workspaces.service import Workspaces  # noqa: E402
from yggdrasil.path.remote_path import RemotePath  # noqa: E402


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _volumes_service(client: MagicMock | None = None) -> MagicMock:
    svc = MagicMock(spec=Volumes)
    svc.client = client or MagicMock()
    return svc


def _workspaces_service(client: MagicMock | None = None) -> MagicMock:
    svc = MagicMock(spec=Workspaces)
    svc.client = client or MagicMock()
    return svc


def _dbfs_service(client: MagicMock | None = None) -> MagicMock:
    svc = MagicMock(spec=DBFSService)
    svc.client = client or MagicMock()
    return svc


def _volume_round_trip_client(store: dict) -> MagicMock:
    """Mock client that round-trips bytes through the Files API surface."""
    client = MagicMock()
    ws = client.workspace_client.return_value

    def get_metadata(path):
        buf = store.get("buf")
        if buf is None:
            raise FileNotFoundError(path)
        return SimpleNamespace(
            content_length=len(buf),
            content_type=None,
            last_modified=None,
        )

    def get_directory_metadata(path):
        raise FileNotFoundError(path)

    def download(path):
        buf = store.get("buf")
        if buf is None:
            raise FileNotFoundError(path)
        return SimpleNamespace(
            contents=SimpleNamespace(read=lambda: buf),
            content_type=None,
            last_modified=None,
        )

    def upload(*, file_path, contents, overwrite):
        store["buf"] = (
            bytes(contents)
            if isinstance(contents, (bytes, bytearray, memoryview))
            else contents.read()
        )

    ws.files.get_metadata.side_effect = get_metadata
    ws.files.get_directory_metadata.side_effect = get_directory_metadata
    ws.files.download.side_effect = download
    ws.files.upload.side_effect = upload
    return client


@pytest.fixture(autouse=True)
def _clear_singletons():
    RemotePath._INSTANCES.clear()
    yield
    RemotePath._INSTANCES.clear()


# ---------------------------------------------------------------------------
# VolumePath — construction
# ---------------------------------------------------------------------------


class TestVolumePathConstruction:

    def test_from_posix_path(self):
        svc = _volumes_service()
        p = VolumePath("/Volumes/cat/sch/vol/data/file.parquet", service=svc)
        assert isinstance(p, VolumePath)
        assert p.catalog_name == "cat"
        assert p.schema_name == "sch"
        assert p.volume_name == "vol"
        assert "cat" in p.full_path()
        assert "vol" in p.full_path()

    def test_from_dbfs_url(self):
        svc = _volumes_service()
        p = VolumePath("dbfs:///Volumes/cat/sch/vol/file.txt", service=svc)
        assert p.catalog_name == "cat"
        assert p.volume_name == "vol"

    def test_api_path_starts_with_volumes(self):
        svc = _volumes_service()
        p = VolumePath("/Volumes/cat/sch/vol/sub/file.csv", service=svc)
        assert p.api_path.startswith("/Volumes/")
        assert "cat" in p.api_path

    def test_full_path(self):
        svc = _volumes_service()
        p = VolumePath("/Volumes/cat/sch/vol/deep/nested/file.csv", service=svc)
        fp = p.full_path()
        assert fp.startswith("/Volumes/")
        assert "cat/sch/vol/deep/nested/file.csv" in fp


# ---------------------------------------------------------------------------
# VolumePath — mock read / write / mkdir / remove
# ---------------------------------------------------------------------------


class TestVolumePathMockReadWrite:

    def test_read_calls_files_download_once(self):
        content = b"hello volume"
        client = MagicMock()
        ws = client.workspace_client.return_value
        ws.files.download.return_value = SimpleNamespace(
            contents=SimpleNamespace(read=lambda: content),
            content_type=None,
            last_modified=None,
        )
        svc = _volumes_service(client)
        p = VolumePath("/Volumes/cat/sch/vol/test.bin", service=svc)
        data = bytes(p._read_mv(len(content), 0))
        ws.files.download.assert_called_once()
        assert data == content

    def test_write_calls_files_upload(self):
        store = {}
        client = _volume_round_trip_client(store)
        svc = _volumes_service(client)
        p = VolumePath("/Volumes/cat/sch/vol/out.bin", service=svc)
        p.write_bytes(b"written", overwrite=True)
        ws = client.workspace_client.return_value
        ws.files.upload.assert_called()
        assert store["buf"] == b"written"

    def test_mkdir_calls_create_directory(self):
        client = MagicMock()
        svc = _volumes_service(client)
        p = VolumePath("/Volumes/cat/sch/vol/newdir/", service=svc)
        p._mkdir(parents=True, exist_ok=True)
        ws = client.workspace_client.return_value
        ws.files.create_directory.assert_called()

    def test_remove_calls_files_delete(self):
        client = MagicMock()
        svc = _volumes_service(client)
        p = VolumePath("/Volumes/cat/sch/vol/del.bin", service=svc)
        p._remove_file(missing_ok=True, wait=None)
        ws = client.workspace_client.return_value
        ws.files.delete.assert_called()


class TestVolumePathRoundTrip:

    def test_write_read_bytes_roundtrip(self):
        store = {}
        client = _volume_round_trip_client(store)
        svc = _volumes_service(client)
        payload = b"roundtrip data!"
        p = VolumePath("/Volumes/cat/sch/vol/rt.bin", service=svc)
        p.write_bytes(payload, overwrite=True)

        p.invalidate_singleton()
        p2 = VolumePath("/Volumes/cat/sch/vol/rt.bin", service=svc)
        got = bytes(p2.read_bytes())
        assert got == payload

    def test_size_after_write(self):
        store = {}
        client = _volume_round_trip_client(store)
        svc = _volumes_service(client)
        p = VolumePath("/Volumes/cat/sch/vol/sized.bin", service=svc)
        p.write_bytes(b"x" * 42, overwrite=True)
        assert p.size == 42


# ---------------------------------------------------------------------------
# DBFSPath — construction
# ---------------------------------------------------------------------------


class TestDBFSPathConstruction:

    def test_from_dbfs_url(self):
        svc = _dbfs_service()
        p = DBFSPath("dbfs+dbfs:///tmp/data.parquet", service=svc)
        assert isinstance(p, DBFSPath)
        assert p.full_path() == "/dbfs/tmp/data.parquet"

    def test_from_posix_path(self):
        svc = _dbfs_service()
        p = DBFSPath("/dbfs/mnt/delta/table", service=svc)
        assert p.api_path == "/mnt/delta/table"
        assert p.full_path() == "/dbfs/mnt/delta/table"

    def test_api_path_strips_dbfs_prefix(self):
        svc = _dbfs_service()
        p = DBFSPath("dbfs:/mnt/data/table", service=svc)
        assert p.api_path.startswith("/")


# ---------------------------------------------------------------------------
# DBFSPath — mock read / mkdir
# ---------------------------------------------------------------------------


class TestDBFSPathMockReadWrite:

    def test_read_calls_dbfs_read(self):
        raw = b"dbfs content"
        encoded = base64.b64encode(raw).decode()
        client = MagicMock()
        ws = client.workspace_client.return_value
        ws.dbfs.read.return_value = SimpleNamespace(
            data=encoded,
            bytes_read=len(raw),
        )
        svc = _dbfs_service(client)
        p = DBFSPath("/dbfs/test/read.bin", service=svc)
        data = bytes(p._read_mv(100, 0))
        ws.dbfs.read.assert_called()
        assert data == raw

    def test_mkdir_calls_dbfs_mkdirs(self):
        client = MagicMock()
        svc = _dbfs_service(client)
        p = DBFSPath("/dbfs/test/newdir/", service=svc)
        p._mkdir(parents=True, exist_ok=True)
        ws = client.workspace_client.return_value
        ws.dbfs.mkdirs.assert_called()

    def test_remove_calls_dbfs_delete(self):
        client = MagicMock()
        svc = _dbfs_service(client)
        p = DBFSPath("/dbfs/test/del.bin", service=svc)
        p._remove_file(missing_ok=True, wait=None)
        ws = client.workspace_client.return_value
        ws.dbfs.delete.assert_called()


# ---------------------------------------------------------------------------
# WorkspacePath — construction
# ---------------------------------------------------------------------------


class TestWorkspacePathConstruction:

    def test_from_workspace_url(self):
        svc = _workspaces_service()
        p = WorkspacePath("dbfs+workspace:///Users/me/notebook", service=svc)
        assert isinstance(p, WorkspacePath)
        assert "/Workspace/" in p.full_path()

    def test_from_posix_path(self):
        svc = _workspaces_service()
        p = WorkspacePath("/Workspace/Users/someone/dir/file.py", service=svc)
        fp = p.full_path()
        assert fp.startswith("/Workspace/")
        assert "Users/someone/dir/file.py" in fp


# ---------------------------------------------------------------------------
# WorkspacePath — mock read / write / mkdir
# ---------------------------------------------------------------------------


class TestWorkspacePathMockReadWrite:

    def test_read_calls_workspace_download(self):
        content = b"workspace content"
        client = MagicMock()
        ws = client.workspace_client.return_value
        ws.workspace.download.return_value = SimpleNamespace(
            contents=SimpleNamespace(read=lambda: content),
            content_type=None,
            last_modified=None,
        )
        svc = _workspaces_service(client)
        p = WorkspacePath("/Workspace/test/file.py", service=svc)
        data = bytes(p._read_mv(100, 0))
        ws.workspace.download.assert_called()
        assert data == content

    def test_write_calls_workspace_upload(self):
        client = MagicMock()
        svc = _workspaces_service(client)
        p = WorkspacePath("/Workspace/test/out.py", service=svc)
        p._write_mv(memoryview(b"print('hello')"), 0)
        ws = client.workspace_client.return_value
        ws.workspace.upload.assert_called()

    def test_mkdir_calls_workspace_mkdirs(self):
        client = MagicMock()
        svc = _workspaces_service(client)
        p = WorkspacePath("/Workspace/test/newdir", service=svc)
        p._mkdir(parents=True, exist_ok=True)
        ws = client.workspace_client.return_value
        ws.workspace.mkdirs.assert_called()

    def test_remove_calls_workspace_delete(self):
        client = MagicMock()
        svc = _workspaces_service(client)
        p = WorkspacePath("/Workspace/test/del.py", service=svc)
        p._remove_file(missing_ok=True, wait=None)
        ws = client.workspace_client.return_value
        ws.workspace.delete.assert_called()


# ---------------------------------------------------------------------------
# Singleton caching
# ---------------------------------------------------------------------------


class TestDatabricksPathSingleton:

    def test_same_url_same_instance(self):
        svc = _volumes_service()
        a = VolumePath("/Volumes/cat/sch/vol/file.txt", service=svc)
        b = VolumePath("/Volumes/cat/sch/vol/file.txt", service=svc)
        assert a is b

    def test_different_url_different_instance(self):
        svc = _volumes_service()
        a = VolumePath("/Volumes/cat/sch/vol/file_a.txt", service=svc)
        b = VolumePath("/Volumes/cat/sch/vol/file_b.txt", service=svc)
        assert a is not b

    def test_dbfs_singleton(self):
        svc = _dbfs_service()
        a = DBFSPath("/dbfs/test/same.bin", service=svc)
        b = DBFSPath("/dbfs/test/same.bin", service=svc)
        assert a is b

    def test_workspace_singleton(self):
        svc = _workspaces_service()
        a = WorkspacePath("/Workspace/test/same.py", service=svc)
        b = WorkspacePath("/Workspace/test/same.py", service=svc)
        assert a is b


# ---------------------------------------------------------------------------
# Integration tests — require DATABRICKS_HOST env var
# ---------------------------------------------------------------------------


_HAS_WORKSPACE = bool(os.getenv("DATABRICKS_HOST"))


@pytest.mark.skipif(not _HAS_WORKSPACE, reason="DATABRICKS_HOST not set")
class TestVolumePathIntegration:

    @pytest.fixture(autouse=True)
    def _setup(self):
        from yggdrasil.databricks import DatabricksClient
        self.client = DatabricksClient.current()
        self.catalog = os.getenv("DATABRICKS_CATALOG", "main")
        self.schema = os.getenv("DATABRICKS_SCHEMA", "default")
        self.volume = os.getenv("DATABRICKS_VOLUME", "ygg_test")

    def _volume_path(self, path):
        RemotePath._INSTANCES.clear()
        return VolumePath(
            f"/Volumes/{self.catalog}/{self.schema}/{self.volume}/{path}",
            client=self.client,
        )

    def test_write_read_bytes_roundtrip(self):
        p = self._volume_path("_ygg_test/integration.bin")
        p.write_bytes(b"integration test data")
        p.invalidate_singleton(remove_global=False)
        data = p.read_bytes()
        assert data == b"integration test data"
        p._remove_file(missing_ok=True, wait=None)

    def test_write_read_arrow_roundtrip(self):
        import pyarrow as pa
        p = self._volume_path("_ygg_test/integration.parquet")
        leaf = p.as_media("parquet")
        table = pa.table({"x": [1, 2, 3]})
        leaf.write_arrow_table(table)
        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        p._remove_file(missing_ok=True, wait=None)

    def test_mkdir_and_list(self):
        p = self._volume_path("_ygg_test/subdir/")
        p._mkdir(parents=True, exist_ok=True)
        parent = self._volume_path("_ygg_test/")
        entries = list(parent._ls())
        assert any("subdir" in str(e) for e in entries)

    def test_stat_returns_size(self):
        p = self._volume_path("_ygg_test/stat.bin")
        p.write_bytes(b"x" * 100)
        p.invalidate_singleton(remove_global=False)
        assert p.size >= 100
        p._remove_file(missing_ok=True, wait=None)
