"""DatabricksPath construction, read/write, and singleton tests using stubs."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

pytest.importorskip("databricks.sdk", reason="databricks-sdk not installed")

from yggdrasil.databricks.fs import DBFSPath, VolumePath, WorkspacePath  # noqa: E402
from yggdrasil.databricks.fs.service import DBFSService  # noqa: E402
from yggdrasil.databricks.path import DatabricksPath  # noqa: E402
from yggdrasil.databricks.volume.volumes import Volumes  # noqa: E402
from yggdrasil.databricks.workspaces.service import Workspaces  # noqa: E402
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile  # noqa: E402
from yggdrasil.path.remote_path import RemotePath  # noqa: E402


# ---------------------------------------------------------------------------
# Stub helpers — mock service + workspace client for each surface
# ---------------------------------------------------------------------------


def _volumes_service(client: MagicMock) -> MagicMock:
    svc = MagicMock(spec=Volumes)
    svc.client = client
    return svc


def _workspaces_service(client: MagicMock) -> MagicMock:
    svc = MagicMock(spec=Workspaces)
    svc.client = client
    return svc


def _dbfs_service(client: MagicMock) -> MagicMock:
    svc = MagicMock(spec=DBFSService)
    svc.client = client
    return svc


def _volume_round_trip_client(store: dict) -> MagicMock:
    """Mock DatabricksClient that round-trips bytes through the Files API."""
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
def _clear_singleton_cache():
    RemotePath._INSTANCES.clear()
    yield
    RemotePath._INSTANCES.clear()


# ===========================================================================
# TestVolumePathConstruction
# ===========================================================================


class TestVolumePathConstruction:

    def test_from_dbfs_url(self):
        client = MagicMock()
        svc = _volumes_service(client)
        vp = VolumePath("dbfs:///Volumes/cat/sch/vol/file.txt", service=svc)
        assert isinstance(vp, VolumePath)
        assert vp.catalog_name == "cat"
        assert vp.schema_name == "sch"
        assert vp.volume_name == "vol"

    def test_catalog_schema_volume_parsing(self):
        client = MagicMock()
        svc = _volumes_service(client)
        vp = VolumePath("/Volumes/my_cat/my_sch/my_vol/subdir/data.parquet", service=svc)
        assert vp.catalog_name == "my_cat"
        assert vp.schema_name == "my_sch"
        assert vp.volume_name == "my_vol"

    def test_full_path_includes_volumes_prefix(self):
        client = MagicMock()
        svc = _volumes_service(client)
        vp = VolumePath("/Volumes/cat/sch/vol/deep/nested/file.csv", service=svc)
        fp = vp.full_path()
        assert fp.startswith("/Volumes/")
        assert "cat/sch/vol/deep/nested/file.csv" in fp


# ===========================================================================
# TestVolumePathReadWrite
# ===========================================================================


class TestVolumePathReadWrite:

    def test_write_bytes_read_back(self):
        store = {}
        client = _volume_round_trip_client(store)
        svc = _volumes_service(client)
        vp = VolumePath("/Volumes/cat/sch/vol/hello.bin", service=svc)

        payload = b"hello, databricks!"
        vp.write_bytes(payload, overwrite=True)
        assert store["buf"] == payload

        vp.invalidate_singleton()
        vp2 = VolumePath("/Volumes/cat/sch/vol/hello.bin", service=svc)
        got = bytes(vp2.read_bytes())
        assert got == payload

    def test_write_arrow_via_as_media(self):
        store = {}
        client = _volume_round_trip_client(store)
        svc = _volumes_service(client)
        vp = VolumePath("/Volumes/cat/sch/vol/data.arrow", service=svc)

        table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        media = ArrowIPCFile(holder=vp, owns_holder=False)
        media.write_arrow_table(table)

        assert len(store["buf"]) > 0
        # Arrow IPC files start with "ARROW1" magic
        assert store["buf"][:6] == b"ARROW1"

    def test_size_after_write(self):
        store = {}
        client = _volume_round_trip_client(store)
        svc = _volumes_service(client)
        vp = VolumePath("/Volumes/cat/sch/vol/sized.bin", service=svc)

        payload = b"x" * 42
        vp.write_bytes(payload, overwrite=True)
        assert vp.size == 42


# ===========================================================================
# TestWorkspacePathConstruction
# ===========================================================================


class TestWorkspacePathConstruction:

    def test_from_workspace_url(self):
        client = MagicMock()
        svc = _workspaces_service(client)
        wp = WorkspacePath("dbfs+workspace:///Users/me/notebook", service=svc)
        assert isinstance(wp, WorkspacePath)
        assert "/Workspace/" in wp.full_path()

    def test_path_parsing(self):
        client = MagicMock()
        svc = _workspaces_service(client)
        wp = WorkspacePath("/Workspace/Users/someone/dir/file.py", service=svc)
        fp = wp.full_path()
        assert fp.startswith("/Workspace/")
        assert "Users/someone/dir/file.py" in fp


# ===========================================================================
# TestDBFSPathConstruction
# ===========================================================================


class TestDBFSPathConstruction:

    def test_from_dbfs_url(self):
        client = MagicMock()
        svc = _dbfs_service(client)
        dp = DBFSPath("dbfs+dbfs:///tmp/data.parquet", service=svc)
        assert isinstance(dp, DBFSPath)
        assert dp.full_path() == "/dbfs/tmp/data.parquet"

    def test_path_parsing_from_posix(self):
        client = MagicMock()
        svc = _dbfs_service(client)
        dp = DBFSPath("/dbfs/mnt/delta/table", service=svc)
        assert dp.api_path == "/mnt/delta/table"
        assert dp.full_path() == "/dbfs/mnt/delta/table"


# ===========================================================================
# TestDatabricksPathSingleton
# ===========================================================================


class TestDatabricksPathSingleton:

    def test_same_url_same_instance(self):
        client = MagicMock()
        svc = _volumes_service(client)
        a = VolumePath("/Volumes/cat/sch/vol/file.txt", service=svc)
        b = VolumePath("/Volumes/cat/sch/vol/file.txt", service=svc)
        assert a is b

    def test_different_url_different_instance(self):
        client = MagicMock()
        svc = _volumes_service(client)
        a = VolumePath("/Volumes/cat/sch/vol/file_a.txt", service=svc)
        b = VolumePath("/Volumes/cat/sch/vol/file_b.txt", service=svc)
        assert a is not b
