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
from yggdrasil.io.primitive.arrow_ipc_io import ArrowIPCIO
from yggdrasil.io.primitive.csv_io import CsvIO
from yggdrasil.io.primitive.ndjson_io import NDJsonIO
from yggdrasil.io.primitive.parquet_io import ParquetIO


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


@pytest.fixture(autouse=True)
def reset_remote_stat_cache():
    from yggdrasil.io.path.remote_path import RemotePath
    RemotePath._STAT_CACHE.clear()
    yield
    RemotePath._STAT_CACHE.clear()


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


def _dbfs_round_trip_workspace(payload_holder: dict) -> MagicMock:
    """Mock workspace that round-trips bytes through DBFS."""
    ws = MagicMock()

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
    return ws


# ---------------------------------------------------------------------------
# Parquet over S3
# ---------------------------------------------------------------------------


class TestParquetOverS3:

    def test_round_trip(self, table) -> None:
        store = {}
        client = _s3_round_trip_client(store)
        s3 = S3Path("s3://my-bucket/data.parquet", client=client)

        writer = ParquetIO(holder=s3, owns_holder=False)
        writer.write_arrow_table(table)
        # Bytes ended up in the mock store.
        assert store["buf"].startswith(b"PAR1")

        reader = ParquetIO(holder=s3, owns_holder=False)
        loaded = reader.read_arrow_table()
        assert loaded.equals(table)

    def test_collect_schema(self, table) -> None:
        store = {}
        client = _s3_round_trip_client(store)
        s3 = S3Path("s3://my-bucket/data.parquet", client=client)
        ParquetIO(holder=s3, owns_holder=False).write_arrow_table(table)

        schema = ParquetIO(holder=s3, owns_holder=False).collect_schema()
        assert set(schema.field_names()) == {"id", "name"}


# ---------------------------------------------------------------------------
# CSV over S3
# ---------------------------------------------------------------------------


class TestCsvOverS3:

    def test_round_trip(self, table) -> None:
        store = {}
        client = _s3_round_trip_client(store)
        s3 = S3Path("s3://my-bucket/data.csv", client=client)

        CsvIO(holder=s3, owns_holder=False).write_arrow_table(table)
        loaded = CsvIO(holder=s3, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Arrow IPC over S3
# ---------------------------------------------------------------------------


class TestArrowIPCOverS3:

    def test_round_trip(self, table) -> None:
        store = {}
        client = _s3_round_trip_client(store)
        s3 = S3Path("s3://my-bucket/data.arrow", client=client)

        ArrowIPCIO(holder=s3, owns_holder=False).write_arrow_table(table)
        loaded = ArrowIPCIO(holder=s3, owns_holder=False).read_arrow_table()
        assert loaded.equals(table)


# ---------------------------------------------------------------------------
# NDJSON over S3
# ---------------------------------------------------------------------------


class TestNDJsonOverS3:

    def test_round_trip(self, table) -> None:
        store = {}
        client = _s3_round_trip_client(store)
        s3 = S3Path("s3://my-bucket/data.ndjson", client=client)

        NDJsonIO(holder=s3, owns_holder=False).write_arrow_table(table)
        # Sanity check the line shape on the wire.
        lines = store["buf"].decode("utf-8").splitlines()
        assert len(lines) == 3
        loaded = NDJsonIO(holder=s3, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Parquet over DBFS
# ---------------------------------------------------------------------------


class TestParquetOverDBFS:

    def test_round_trip(self, table) -> None:
        store = {}
        ws = _dbfs_round_trip_workspace(store)
        dbfs = DBFSPath("/dbfs/data.parquet", workspace=ws)

        ParquetIO(holder=dbfs, owns_holder=False).write_arrow_table(table)
        assert store["buf"].startswith(b"PAR1")

        # Stat cache is invalidated by the write; the reader's first
        # head call sees the freshly committed size.
        from yggdrasil.io.path.remote_path import RemotePath
        RemotePath._STAT_CACHE.clear()
        loaded = ParquetIO(holder=dbfs, owns_holder=False).read_arrow_table()
        assert loaded.equals(table)


# ---------------------------------------------------------------------------
# CSV over DBFS
# ---------------------------------------------------------------------------


class TestCsvOverDBFS:

    def test_round_trip(self, table) -> None:
        store = {}
        ws = _dbfs_round_trip_workspace(store)
        dbfs = DBFSPath("/dbfs/data.csv", workspace=ws)

        CsvIO(holder=dbfs, owns_holder=False).write_arrow_table(table)
        from yggdrasil.io.path.remote_path import RemotePath
        RemotePath._STAT_CACHE.clear()
        loaded = CsvIO(holder=dbfs, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Arrow IPC over Volume (Files API)
# ---------------------------------------------------------------------------


class TestArrowIPCOverVolume:

    def test_round_trip(self, table) -> None:
        store = {}
        ws = MagicMock()

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
            store["buf"] = contents.getvalue()

        ws.files.get_metadata.side_effect = get_metadata
        ws.files.get_directory_metadata.side_effect = get_directory_metadata
        ws.files.download.side_effect = download
        ws.files.upload.side_effect = upload

        vol = VolumePath("/Volumes/c/s/v/data.arrow", workspace=ws)
        ArrowIPCIO(holder=vol, owns_holder=False).write_arrow_table(table)
        from yggdrasil.io.path.remote_path import RemotePath
        RemotePath._STAT_CACHE.clear()
        loaded = ArrowIPCIO(holder=vol, owns_holder=False).read_arrow_table()
        assert loaded.equals(table)
