"""Tests for Databricks path types: VolumePath, DBFSPath, WorkspacePath.

Mock tests verify minimum SDK calls. Integration tests (marked
@pytest.mark.integration) run against a live workspace when
DATABRICKS_HOST is set.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import io

import pytest

try:
    from databricks.sdk.service.files import (
        DirectoryEntry,
        DownloadResponse,
    )
    from databricks.sdk.service.workspace import (
        ObjectInfo,
        ObjectType,
    )
    HAS_SDK = True
except ImportError:
    HAS_SDK = False

pytestmark = pytest.mark.skipif(not HAS_SDK, reason="databricks-sdk not installed")


def _mock_client(host="https://test.cloud.databricks.com"):
    client = MagicMock()
    client.base_url = MagicMock()
    client.base_url.to_string.return_value = host
    client.base_url.host = "test.cloud.databricks.com"
    wc = MagicMock()
    client.workspace_client.return_value = wc
    return client, wc


# ---------------------------------------------------------------------------
# VolumePath mock tests
# ---------------------------------------------------------------------------


class TestVolumePathConstruction:

    def test_from_volumes_url(self):
        from yggdrasil.databricks.fs.volume_path import VolumePath
        client, _ = _mock_client()
        VolumePath._INSTANCES.clear()
        p = VolumePath(
            catalog_name="cat",
            schema_name="sch",
            volume_name="vol",
            path="data/file.parquet",
            client=client,
        )
        assert "cat" in p.full_path()
        assert "vol" in p.full_path()

    def test_api_path(self):
        from yggdrasil.databricks.fs.volume_path import VolumePath
        client, _ = _mock_client()
        VolumePath._INSTANCES.clear()
        p = VolumePath(
            catalog_name="cat",
            schema_name="sch",
            volume_name="vol",
            path="sub/file.csv",
            client=client,
        )
        assert p.api_path.startswith("/Volumes/")
        assert "cat" in p.api_path


class TestVolumePathMockReadWrite:

    def test_read_calls_files_download_once(self):
        from yggdrasil.databricks.fs.volume_path import VolumePath
        client, wc = _mock_client()
        VolumePath._INSTANCES.clear()

        content = b"hello volume"
        resp = MagicMock()
        resp.contents = io.BytesIO(content)
        resp.read.return_value = content
        wc.files.download.return_value = resp

        p = VolumePath(
            catalog_name="cat", schema_name="sch", volume_name="vol",
            path="test.bin", client=client,
        )
        data = bytes(p._read_mv(len(content), 0))
        wc.files.download.assert_called_once()
        assert data == content

    def test_write_calls_files_upload_once(self):
        from yggdrasil.databricks.fs.volume_path import VolumePath
        client, wc = _mock_client()
        VolumePath._INSTANCES.clear()

        p = VolumePath(
            catalog_name="cat", schema_name="sch", volume_name="vol",
            path="out.bin", client=client,
        )
        p._write_mv(memoryview(b"written"), 0)
        wc.files.upload.assert_called_once()

    def test_mkdir_calls_create_directory(self):
        from yggdrasil.databricks.fs.volume_path import VolumePath
        client, wc = _mock_client()
        VolumePath._INSTANCES.clear()

        p = VolumePath(
            catalog_name="cat", schema_name="sch", volume_name="vol",
            path="newdir/", client=client,
        )
        p._mkdir(parents=True, exist_ok=True)
        wc.files.create_directory.assert_called()

    def test_remove_calls_files_delete(self):
        from yggdrasil.databricks.fs.volume_path import VolumePath
        client, wc = _mock_client()
        VolumePath._INSTANCES.clear()

        p = VolumePath(
            catalog_name="cat", schema_name="sch", volume_name="vol",
            path="del.bin", client=client,
        )
        p._remove_file(missing_ok=True, wait=None)
        wc.files.delete.assert_called()


# ---------------------------------------------------------------------------
# DBFSPath mock tests
# ---------------------------------------------------------------------------


class TestDBFSPathConstruction:

    def test_from_dbfs_url(self):
        from yggdrasil.databricks.fs.dbfs_path import DBFSPath
        client, _ = _mock_client()
        DBFSPath._INSTANCES.clear()
        p = DBFSPath(data="dbfs:/test/file.parquet", client=client)
        assert "test" in p.full_path()

    def test_api_path_strips_dbfs_prefix(self):
        from yggdrasil.databricks.fs.dbfs_path import DBFSPath
        client, _ = _mock_client()
        DBFSPath._INSTANCES.clear()
        p = DBFSPath(data="dbfs:/mnt/data/table", client=client)
        assert p.api_path.startswith("/")


class TestDBFSPathMockReadWrite:

    def test_read_calls_dbfs_read_once(self):
        from yggdrasil.databricks.fs.dbfs_path import DBFSPath
        client, wc = _mock_client()
        DBFSPath._INSTANCES.clear()

        resp = MagicMock()
        resp.data = b"dbfs content"
        resp.bytes_read = len(resp.data)
        wc.dbfs.read.return_value = resp

        p = DBFSPath(data="dbfs:/test/read.bin", client=client)
        data = bytes(p._read_mv(100, 0))
        wc.dbfs.read.assert_called()

    def test_mkdir_calls_dbfs_mkdirs(self):
        from yggdrasil.databricks.fs.dbfs_path import DBFSPath
        client, wc = _mock_client()
        DBFSPath._INSTANCES.clear()

        p = DBFSPath(data="dbfs:/test/newdir/", client=client)
        p._mkdir(parents=True, exist_ok=True)
        wc.dbfs.mkdirs.assert_called()


# ---------------------------------------------------------------------------
# WorkspacePath mock tests
# ---------------------------------------------------------------------------


class TestWorkspacePathConstruction:

    def test_from_workspace_url(self):
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath
        client, _ = _mock_client()
        WorkspacePath._INSTANCES.clear()
        p = WorkspacePath(data="/Workspace/test/notebook", client=client)
        assert "test" in p.full_path()


class TestWorkspacePathMockReadWrite:

    def test_read_calls_workspace_download(self):
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath
        client, wc = _mock_client()
        WorkspacePath._INSTANCES.clear()

        resp = MagicMock()
        resp.read.return_value = b"workspace content"
        wc.workspace.download.return_value = resp

        p = WorkspacePath(data="/Workspace/test/file.py", client=client)
        data = bytes(p._read_mv(100, 0))
        wc.workspace.download.assert_called()

    def test_write_calls_workspace_upload(self):
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath
        client, wc = _mock_client()
        WorkspacePath._INSTANCES.clear()

        p = WorkspacePath(data="/Workspace/test/out.py", client=client)
        p._write_mv(memoryview(b"print('hello')"), 0)
        wc.workspace.upload.assert_called()

    def test_mkdir_calls_workspace_mkdirs(self):
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath
        client, wc = _mock_client()
        WorkspacePath._INSTANCES.clear()

        p = WorkspacePath(data="/Workspace/test/newdir", client=client)
        p._mkdir(parents=True, exist_ok=True)
        wc.workspace.mkdirs.assert_called()


# ---------------------------------------------------------------------------
# Integration tests — require DATABRICKS_HOST env var
# ---------------------------------------------------------------------------


import os
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
        from yggdrasil.databricks.fs.volume_path import VolumePath
        VolumePath._INSTANCES.clear()
        return VolumePath(
            catalog_name=self.catalog, schema_name=self.schema,
            volume_name=self.volume, path=path, client=self.client,
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
