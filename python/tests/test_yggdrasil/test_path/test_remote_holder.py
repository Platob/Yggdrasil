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

from tests.test_yggdrasil.test_databricks._files_fake import wire_files_session
from tests.test_yggdrasil.test_aws._fake_s3 import FakeS3, wire_s3_path, reset_s3_singletons
from yggdrasil.io.arrow_ipc_file import ArrowIPCFile
from yggdrasil.io.csv_file import CSVFile
from yggdrasil.io.ndjson_file import NDJSONFile
from yggdrasil.io.parquet_file import ParquetFile


def _s3(url: str = "s3://my-bucket/data.bin"):
    """A real (pure-HTTP) :class:`S3Path` backed by an in-memory FakeS3.

    Returns ``(path, fake)``; ``fake.objects`` is the ``{key: bytes}`` store
    and ``fake.calls`` counts the REST primitives (``put`` / ``get`` /
    ``head`` / ``delete`` / multipart) the path issued.
    """
    reset_s3_singletons()
    fake = FakeS3()
    return wire_s3_path(fake, url, bucket="my-bucket"), fake


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
    # ``VolumePath`` file ops route over the Files-API HTTP seam; wire a
    # fake session that translates back onto ``workspace.files``.
    svc.client = wire_files_session(client)
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
        s3, fake = _s3("s3://my-bucket/data.parquet")
        ParquetFile(holder=s3, owns_holder=False).write_arrow_table(table)
        assert fake.objects["data.parquet"].startswith(b"PAR1")

        loaded = ParquetFile(holder=s3, owns_holder=False).read_arrow_table()
        assert loaded.equals(table)

    def test_collect_schema(self, table) -> None:
        s3, _ = _s3("s3://my-bucket/data.parquet")
        ParquetFile(holder=s3, owns_holder=False).write_arrow_table(table)
        schema = ParquetFile(holder=s3, owns_holder=False).collect_schema()
        assert set(schema.field_names()) == {"id", "name"}


# ---------------------------------------------------------------------------
# CSV over S3
# ---------------------------------------------------------------------------


class TestCsvOverS3:

    def test_round_trip(self, table) -> None:
        s3, fake = _s3("s3://my-bucket/data.csv")

        CSVFile(holder=s3, owns_holder=False).write_arrow_table(table)
        loaded = CSVFile(holder=s3, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Arrow IPC over S3
# ---------------------------------------------------------------------------


class TestArrowIPCOverS3:

    def test_round_trip(self, table) -> None:
        s3, fake = _s3("s3://my-bucket/data.arrow")

        ArrowIPCFile(holder=s3, owns_holder=False).write_arrow_table(table)
        loaded = ArrowIPCFile(holder=s3, owns_holder=False).read_arrow_table()
        assert loaded.equals(table)


# ---------------------------------------------------------------------------
# NDJSON over S3
# ---------------------------------------------------------------------------


class TestNDJsonOverS3:

    def test_round_trip(self, table) -> None:
        s3, fake = _s3("s3://my-bucket/data.ndjson")

        NDJSONFile(holder=s3, owns_holder=False).write_arrow_table(table)
        # Sanity check the line shape on the wire.
        lines = fake.objects["data.ndjson"].decode("utf-8").splitlines()
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


class TestS3WriteColdPath:
    """A cold remote path (no prior stat) writes in one signed PUT — no
    speculative HEAD/GET probe first."""

    def test_write_all_one_call(self, table) -> None:
        s3, fake = _s3()
        s3.write_bytes(b"hello", overwrite=True)
        assert fake.calls.get("put") == 1
        assert fake.calls.get("head", 0) == 0
        assert fake.calls.get("get", 0) == 0
        assert fake.objects["data.bin"] == b"hello"

    def test_cursor_write_one_call(self, table) -> None:
        s3, fake = _s3()
        with s3.open("wb") as f:
            f.write(b"cursor-data")
        assert fake.calls.get("put") == 1
        assert fake.calls.get("head", 0) == 0
        assert fake.calls.get("get", 0) == 0
        assert fake.objects["data.bin"] == b"cursor-data"

    def test_cursor_seek_write_one_call(self, table) -> None:
        s3, fake = _s3()
        with s3.open("wb") as f:
            f.write(b"head")
            f.seek(10)
            f.write(b"tail")
        assert fake.calls.get("put") == 1
        assert fake.calls.get("get", 0) == 0
        buf = fake.objects["data.bin"]
        assert len(buf) == 14 and buf[:4] == b"head" and buf[10:] == b"tail"

    def test_parquet_write_one_call(self, table) -> None:
        s3, fake = _s3("s3://my-bucket/data.parquet")
        ParquetFile(holder=s3, owns_holder=False).write_arrow_table(table)
        # Format writes spill + stream → one PUT, no put-then-probe.
        assert fake.calls.get("put") == 1
        assert fake.calls.get("head", 0) == 0
        assert fake.objects["data.parquet"].startswith(b"PAR1")

    def test_arrow_ipc_write_one_call(self, table) -> None:
        s3, fake = _s3("s3://my-bucket/data.arrow")
        ArrowIPCFile(holder=s3, owns_holder=False).write_arrow_table(table)
        assert fake.calls.get("put") == 1
        assert fake.calls.get("head", 0) == 0

    def test_csv_write_one_call(self, table) -> None:
        s3, fake = _s3("s3://my-bucket/data.csv")
        CSVFile(holder=s3, owns_holder=False).write_arrow_table(table)
        assert fake.calls.get("put") == 1
        assert fake.calls.get("head", 0) == 0

    def test_ndjson_write_one_call(self, table) -> None:
        s3, fake = _s3("s3://my-bucket/data.ndjson")
        NDJSONFile(holder=s3, owns_holder=False).write_arrow_table(table)
        assert fake.calls.get("put") == 1
        assert fake.calls.get("head", 0) == 0

    def test_stat_correct_after_write(self, table) -> None:
        s3, _ = _s3()
        s3.write_bytes(b"hello world", overwrite=True)
        assert s3.size == 11
        assert s3.size_known

    def test_read_back_after_cursor_write(self) -> None:
        s3, _ = _s3()
        with s3.open("wb") as f:
            f.write(b"round-trip")
        assert s3.read_bytes() == b"round-trip"
