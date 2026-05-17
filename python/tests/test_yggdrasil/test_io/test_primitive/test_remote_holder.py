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
from yggdrasil.databricks.fs import DBFSPath, VolumePath
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile
from yggdrasil.io.primitive.csv_file import CSVFile
from yggdrasil.io.primitive.ndjson_file import NDJSONFile
from yggdrasil.io.primitive.parquet_file import ParquetFile


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
        s3 = S3Path("s3://my-bucket/data.parquet", client=client)

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
        s3 = S3Path("s3://my-bucket/data.parquet", client=client)
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
        s3 = S3Path("s3://my-bucket/data.csv", client=client)

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
        s3 = S3Path("s3://my-bucket/data.arrow", client=client)

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
        s3 = S3Path("s3://my-bucket/data.ndjson", client=client)

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
        dbfs = DBFSPath("/dbfs/data.parquet", client=client)

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
        dbfs = DBFSPath("/dbfs/data.csv", client=client)

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

        vol = VolumePath("/Volumes/c/s/v/data.arrow", client=client)
        ArrowIPCFile(holder=vol, owns_holder=False).write_arrow_table(table)
        vol.invalidate_singleton()
        loaded = ArrowIPCFile(holder=vol, owns_holder=False).read_arrow_table()
        assert loaded.equals(table)
