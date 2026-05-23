"""Tabular IO over remote (S3 / DBFS) holders, mock-driven.

The Tabular leaves are oblivious to whether their holder is local
or remote — they go through the :class:`Holder` byte primitives.
These tests pin that contract: a parquet / arrow / csv leaf bound
to a mocked S3 holder or DBFS holder round-trips through the SDK
calls without changing the leaf's code path.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from yggdrasil.aws.fs.path import S3Path
from yggdrasil.aws.fs.service import S3Service
from yggdrasil.databricks.fs import DBFSPath, VolumePath
from yggdrasil.databricks.fs.service import DBFSService
from yggdrasil.databricks.volume.volumes import Volumes
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile
from yggdrasil.io.primitive.csv_file import CSVFile
from yggdrasil.io.primitive.ndjson_file import NDJSONFile
from yggdrasil.io.primitive.parquet_file import ParquetFile


def _s3_service(client: MagicMock) -> MagicMock:
    """Wrap a boto-shaped mock client in a mock :class:`S3Service`.

    :class:`S3Path` reaches the boto surface through
    ``self.service.boto_client``.
    """
    svc = MagicMock(spec=S3Service)
    svc.boto_client = client
    return svc


def _dbfs_service(client: MagicMock) -> MagicMock:
    """Wrap a :class:`DatabricksClient`-shaped mock in a mock
    :class:`DBFSService` so :class:`DBFSPath` reaches the workspace
    handle through ``self.service.client``."""
    svc = MagicMock(spec=DBFSService)
    svc.client = client
    return svc


def _volumes_service(client: MagicMock) -> MagicMock:
    """Mock :class:`Volumes` service whose ``client`` is *client*."""
    svc = MagicMock(spec=Volumes)
    svc.client = client
    return svc


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


@pytest.fixture(autouse=True)
def reset_remote_stat_cache():
    yield


# ---------------------------------------------------------------------------
# S3 mock fixtures
# ---------------------------------------------------------------------------


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        pass


def _s3_round_trip_client(payload_holder: dict) -> MagicMock:
    """Return a Mock that round-trips a single object's bytes.

    ``payload_holder`` is a one-key dict (e.g. ``{"buf": b""}``)
    that the client mutates on put_object and reads on get_object,
    so the test can write through the path and read back the same
    bytes via the same mock.
    """
    client = MagicMock()

    def head_object(*, Bucket, Key):
        buf = payload_holder.get("buf")
        if buf is None:
            err = Exception("NoSuchKey")
            err.response = {  # type: ignore[attr-defined]
                "Error": {"Code": "NoSuchKey"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            }
            raise err
        return {"ContentLength": len(buf), "LastModified": None}

    def get_object(*, Bucket, Key, Range=None):
        buf = payload_holder.get("buf") or b""
        if Range:
            # bytes=N-M format; closed range, inclusive end.
            spec = Range.split("=", 1)[1]
            start_str, end_str = spec.split("-", 1)
            start = int(start_str)
            end = int(end_str) if end_str else len(buf) - 1
            return {"Body": _Body(buf[start : end + 1])}
        return {"Body": _Body(buf)}

    def put_object(*, Bucket, Key, Body):
        payload_holder["buf"] = Body if isinstance(Body, (bytes, bytearray)) else bytes(Body)
        return {}

    def list_objects_v2(**kwargs):
        return {"KeyCount": 0}

    client.head_object.side_effect = head_object
    client.get_object.side_effect = get_object
    client.put_object.side_effect = put_object
    client.list_objects_v2.side_effect = list_objects_v2
    return client


# ---------------------------------------------------------------------------
# DBFS mock helpers
# ---------------------------------------------------------------------------


class _DBFSStreamWriter:
    def __init__(self, sink: list[bytes]) -> None:
        self._sink = sink

    def __enter__(self) -> "_DBFSStreamWriter":
        return self

    def __exit__(self, *args) -> None:
        return None

    def write(self, data: bytes) -> None:
        self._sink.append(bytes(data))


def _dbfs_round_trip_client(payload_holder: dict) -> MagicMock:
    """Mock :class:`DatabricksClient` that round-trips bytes through DBFS."""
    client = MagicMock()
    ws = client.workspace_client.return_value

    def get_status(path):
        buf = payload_holder.get("buf")
        if buf is None:
            raise FileNotFoundError(path)
        return SimpleNamespace(
            is_dir=False, file_size=len(buf), modification_time=0,
        )

    def read(*, path, offset, length):
        buf = payload_holder.get("buf") or b""
        return SimpleNamespace(
            data=base64.b64encode(buf[offset : offset + length]).decode(),
        )

    def open_writer(*, path, read=False, write=True, overwrite=True):
        sink: list[bytes] = []
        payload_holder["sink"] = sink
        return _DBFSStreamWriter(sink)

    ws.dbfs.get_status.side_effect = get_status
    ws.dbfs.read.side_effect = read
    ws.dbfs.open.side_effect = open_writer

    def commit_after_write(orig_holder=payload_holder):
        if "sink" in payload_holder:
            payload_holder["buf"] = b"".join(payload_holder["sink"])
            del payload_holder["sink"]

    # Wrap dbfs.open so the sink commits to ``buf`` when the caller
    # exits the context manager. Patch the class so __exit__ commits.
    class _CommittingWriter(_DBFSStreamWriter):
        def __exit__(self, *args) -> None:
            commit_after_write()

    def open_writer_committing(*, path, read=False, write=True, overwrite=True):
        sink: list[bytes] = []
        payload_holder["sink"] = sink
        return _CommittingWriter(sink)

    ws.dbfs.open.side_effect = open_writer_committing
    return client


# ---------------------------------------------------------------------------
# Parquet over S3
# ---------------------------------------------------------------------------


class TestParquetOverS3:

    def test_round_trip(self, table) -> None:
        store = {}
        client = _s3_round_trip_client(store)
        s3 = S3Path("s3://my-bucket/data.parquet", service=_s3_service(client))

        writer = ParquetFile(holder=s3, owns_holder=False)
        writer.write_arrow_table(table)
        # Bytes ended up in the mock store.
        assert store["buf"].startswith(b"PAR1")

        reader = ParquetFile(holder=s3, owns_holder=False)
        loaded = reader.read_arrow_table()
        assert loaded.equals(table)

    def test_collect_schema(self, table) -> None:
        store = {}
        client = _s3_round_trip_client(store)
        s3 = S3Path("s3://my-bucket/data.parquet", service=_s3_service(client))
        ParquetFile(holder=s3, owns_holder=False).write_arrow_table(table)

        schema = ParquetFile(holder=s3, owns_holder=False).collect_schema()
        assert set(schema.field_names()) == {"id", "name"}


# ---------------------------------------------------------------------------
# CSV over S3
# ---------------------------------------------------------------------------


class TestCsvOverS3:

    def test_round_trip(self, table) -> None:
        store = {}
        client = _s3_round_trip_client(store)
        s3 = S3Path("s3://my-bucket/data.csv", service=_s3_service(client))

        CSVFile(holder=s3, owns_holder=False).write_arrow_table(table)
        loaded = CSVFile(holder=s3, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Arrow IPC over S3
# ---------------------------------------------------------------------------


class TestArrowIPCOverS3:

    def test_round_trip(self, table) -> None:
        store = {}
        client = _s3_round_trip_client(store)
        s3 = S3Path("s3://my-bucket/data.arrow", service=_s3_service(client))

        ArrowIPCFile(holder=s3, owns_holder=False).write_arrow_table(table)
        loaded = ArrowIPCFile(holder=s3, owns_holder=False).read_arrow_table()
        assert loaded.equals(table)


# ---------------------------------------------------------------------------
# NDJSON over S3
# ---------------------------------------------------------------------------


class TestNDJsonOverS3:

    def test_round_trip(self, table) -> None:
        store = {}
        client = _s3_round_trip_client(store)
        s3 = S3Path("s3://my-bucket/data.ndjson", service=_s3_service(client))

        NDJSONFile(holder=s3, owns_holder=False).write_arrow_table(table)
        # Sanity check the line shape on the wire.
        lines = store["buf"].decode("utf-8").splitlines()
        assert len(lines) == 3
        loaded = NDJSONFile(holder=s3, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Parquet over DBFS
# ---------------------------------------------------------------------------


class TestParquetOverDBFS:

    def test_round_trip(self, table) -> None:
        store = {}
        client = _dbfs_round_trip_client(store)
        dbfs = DBFSPath("/dbfs/data.parquet", service=_dbfs_service(client))

        ParquetFile(holder=dbfs, owns_holder=False).write_arrow_table(table)
        assert store["buf"].startswith(b"PAR1")

        # Drop the post-write stat cache so the reader's first probe
        # exercises the real ``get_status`` path against the freshly
        # committed buffer.
        dbfs.invalidate_singleton()
        loaded = ParquetFile(holder=dbfs, owns_holder=False).read_arrow_table()
        assert loaded.equals(table)


# ---------------------------------------------------------------------------
# CSV over DBFS
# ---------------------------------------------------------------------------


class TestCsvOverDBFS:

    def test_round_trip(self, table) -> None:
        store = {}
        client = _dbfs_round_trip_client(store)
        dbfs = DBFSPath("/dbfs/data.csv", service=_dbfs_service(client))

        CSVFile(holder=dbfs, owns_holder=False).write_arrow_table(table)
        dbfs.invalidate_singleton()
        loaded = CSVFile(holder=dbfs, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Arrow IPC over Volume (Files API)
# ---------------------------------------------------------------------------


class TestArrowIPCOverVolume:

    def test_round_trip(self, table) -> None:
        store = {}
        client = MagicMock()
        ws = client.workspace_client.return_value

        def get_metadata(path):
            buf = store.get("buf")
            if buf is None:
                raise FileNotFoundError(path)
            return SimpleNamespace(content_length=len(buf), modification_time=0)

        def get_directory_metadata(path):
            raise FileNotFoundError(path)

        def download(path):
            buf = store.get("buf") or b""
            return SimpleNamespace(
                contents=SimpleNamespace(read=lambda: buf),
            )

        def upload(*, file_path, contents, overwrite):
            # ``contents`` is bytes when caller passes a bytes-like
            # payload, otherwise a stream — accept both shapes.
            store["buf"] = (
                bytes(contents)
                if isinstance(contents, (bytes, bytearray, memoryview))
                else contents.read()
            )

        ws.files.get_metadata.side_effect = get_metadata
        ws.files.get_directory_metadata.side_effect = get_directory_metadata
        ws.files.download.side_effect = download
        ws.files.upload.side_effect = upload

        vol = VolumePath("/Volumes/c/s/v/data.arrow", service=_volumes_service(client))
        ArrowIPCFile(holder=vol, owns_holder=False).write_arrow_table(table)
        vol.invalidate_singleton()
        loaded = ArrowIPCFile(holder=vol, owns_holder=False).read_arrow_table()
        assert loaded.equals(table)


# ---------------------------------------------------------------------------
# SDK call-count consistency: cold write = 1 upload, 0 stat probes
# ---------------------------------------------------------------------------


def _s3_call_counts(client: MagicMock) -> dict[str, int]:
    return {
        name: getattr(client, name).call_count
        for name in ("head_object", "get_object", "put_object", "delete_object")
    }


class TestS3WriteColdPath:
    """A cold remote path (no prior stat) must write in 1 SDK call."""

    def _fresh_s3(self, store: dict) -> S3Path:
        from yggdrasil.io.path.remote_path import RemotePath
        RemotePath._INSTANCES.clear()
        client = _s3_round_trip_client(store)
        s3 = S3Path("s3://my-bucket/data.bin", service=_s3_service(client))
        client.head_object.reset_mock()
        client.get_object.reset_mock()
        client.put_object.reset_mock()
        client.delete_object.reset_mock()
        return s3

    def test_write_all_one_call(self, table) -> None:
        store = {}
        s3 = self._fresh_s3(store)
        s3.write_bytes(b"hello", overwrite=True)
        assert s3.service.boto_client.put_object.call_count == 1
        assert s3.service.boto_client.head_object.call_count == 0
        assert s3.service.boto_client.get_object.call_count == 0
        assert store["buf"] == b"hello"

    def test_cursor_write_one_call(self, table) -> None:
        store = {}
        s3 = self._fresh_s3(store)
        with s3.open("wb") as f:
            f.write(b"cursor-data")
        assert s3.service.boto_client.put_object.call_count == 1
        assert s3.service.boto_client.head_object.call_count == 0
        assert s3.service.boto_client.get_object.call_count == 0
        assert store["buf"] == b"cursor-data"

    def test_cursor_seek_write_one_call(self, table) -> None:
        store = {}
        s3 = self._fresh_s3(store)
        with s3.open("wb") as f:
            f.write(b"head")
            f.seek(10)
            f.write(b"tail")
        assert s3.service.boto_client.put_object.call_count == 1
        assert s3.service.boto_client.get_object.call_count == 0
        assert len(store["buf"]) == 14
        assert store["buf"][:4] == b"head"
        assert store["buf"][10:] == b"tail"

    def test_parquet_write_one_call(self, table) -> None:
        store = {}
        s3 = self._fresh_s3(store)
        s3 = S3Path("s3://my-bucket/data.parquet", service=s3.service)
        s3.service.boto_client.put_object.reset_mock()
        ParquetFile(holder=s3, owns_holder=False).write_arrow_table(table)
        assert s3.service.boto_client.put_object.call_count == 1
        assert s3.service.boto_client.head_object.call_count == 0
        assert store["buf"].startswith(b"PAR1")

    def test_arrow_ipc_write_one_call(self, table) -> None:
        store = {}
        s3 = self._fresh_s3(store)
        ArrowIPCFile(holder=s3, owns_holder=False).write_arrow_table(table)
        assert s3.service.boto_client.put_object.call_count == 1
        assert s3.service.boto_client.head_object.call_count == 0

    def test_csv_write_one_call(self, table) -> None:
        store = {}
        s3 = self._fresh_s3(store)
        CSVFile(holder=s3, owns_holder=False).write_arrow_table(table)
        assert s3.service.boto_client.put_object.call_count == 1
        assert s3.service.boto_client.head_object.call_count == 0

    def test_ndjson_write_one_call(self, table) -> None:
        store = {}
        s3 = self._fresh_s3(store)
        NDJSONFile(holder=s3, owns_holder=False).write_arrow_table(table)
        assert s3.service.boto_client.put_object.call_count == 1
        assert s3.service.boto_client.head_object.call_count == 0

    def test_stat_correct_after_write(self, table) -> None:
        store = {}
        s3 = self._fresh_s3(store)
        s3.write_bytes(b"hello world", overwrite=True)
        assert s3.size == 11
        assert s3.size_known

    def test_read_back_after_cursor_write(self) -> None:
        store = {}
        s3 = self._fresh_s3(store)
        with s3.open("wb") as f:
            f.write(b"round-trip")
        data = s3.read_bytes()
        assert data == b"round-trip"
