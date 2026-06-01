"""Tabular format-leaf reads over a streaming holder (the HTTPStream
behind a ``stream=True`` response, and any spilling MemoryStream).

This is the real "do tabular IOs work on streams" question. The leaf
read path (:meth:`ParquetFile.read_arrow_table` etc.) — not a hand-rolled
``pq.read_table(bytes)`` — is what production uses.

Regression covered: a streaming :class:`MemoryStream` reports ``size``
as the bytes pulled *so far*, which is ``0`` before the first read. The
format leaves guard their read with
``if self.size_known and self.size == 0: return`` as an empty-buffer
short-circuit, and the base :class:`Holder.size_known` is ``True`` for
in-memory holders — so on an un-pulled stream the short-circuit fired and
every leaf silently returned **zero rows**. :class:`MemoryStream` now
reports ``size_known`` as ``False`` until EOF, so the short-circuit
defers to the actual read (which drives the pull).

Both the in-window case and the spill-to-disk case (body far larger than
the in-memory window) are exercised, plus a real ``session.get(...,
stream=True)`` end-to-end.
"""
from __future__ import annotations

import http.server
import io
import json
import os
import socket
import threading
import zipfile
from socketserver import ThreadingMixIn

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest

from yggdrasil.http_.session import HTTPSession
from yggdrasil.path.memory import Memory
from yggdrasil.path.memory_stream import MemoryStream
from yggdrasil.io.arrow_ipc_file import ArrowIPCFile
from yggdrasil.io.csv_file import CSVFile
from yggdrasil.io.ndjson_file import NDJSONFile
from yggdrasil.io.parquet_file import ParquetFile


_ROWS = 200_000


@pytest.fixture(scope="module")
def table() -> pa.Table:
    return pa.table({"a": pa.array(range(_ROWS)), "b": ["x" * 16] * _ROWS})


@pytest.fixture(scope="module")
def payloads(table) -> dict[str, bytes]:
    pb = io.BytesIO()
    pq.write_table(table, pb)
    # ArrowIPCFile reads the IPC *file* format (random-access, footer at
    # EOF), not the streaming format — use new_file, not new_stream.
    ab = io.BytesIO()
    with ipc.new_file(ab, table.schema) as w:
        w.write_table(table)
    cb = io.BytesIO()
    pacsv.write_csv(table, cb)
    ndjson = "\n".join(
        json.dumps({"a": i, "b": "x" * 16}) for i in range(_ROWS)
    ).encode()
    return {
        "parquet": pb.getvalue(),
        "arrow": ab.getvalue(),
        "csv": cb.getvalue(),
        "ndjson": ndjson,
    }


_LEAVES = {
    "parquet": ParquetFile,
    "arrow": ArrowIPCFile,
    "csv": CSVFile,
    "ndjson": NDJSONFile,
}


def _spilling(data: bytes, *, window_kib: int = 64) -> MemoryStream:
    """In-memory window far smaller than *data* (so it spills to disk) but
    a retention budget covering the whole body (so nothing is evicted)."""
    return MemoryStream(
        io.BytesIO(data),
        spill_threshold=window_kib * 1024,
        byte_size=len(data) + 4096,
    )


# ---------------------------------------------------------------------------
# The regression itself: un-pulled streaming holder must not look empty
# ---------------------------------------------------------------------------


def test_streaming_holder_size_not_known_until_eof():
    ms = MemoryStream(io.BytesIO(b"abcdef"))
    assert ms.size == 0            # nothing pulled yet
    assert ms.size_known is False  # ...but the body is NOT known-empty
    ms.read_mv(-1, 0)              # pull to EOF
    assert ms.size == 6
    assert ms.size_known is True


def test_empty_streaming_holder_is_known_empty_after_pull():
    ms = MemoryStream(io.BytesIO(b""))
    ms.read_mv(-1, 0)
    assert ms.size == 0
    assert ms.size_known is True


# ---------------------------------------------------------------------------
# Leaf reads over streaming/spilling holders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", list(_LEAVES))
def test_leaf_reads_over_spilled_stream(fmt, payloads):
    data = payloads[fmt]
    assert len(data) > 64 * 1024
    ms = _spilling(data)
    table = _LEAVES[fmt](holder=ms, owns_holder=False).read_arrow_table()
    assert table.num_rows == _ROWS


@pytest.mark.parametrize("fmt", list(_LEAVES))
def test_leaf_reads_over_unspilled_stream(fmt, payloads):
    # Window larger than the body — streams but never spills. Still
    # un-pulled at construction, so this guards the size_known fix
    # independently of the spill path.
    data = payloads[fmt]
    ms = MemoryStream(io.BytesIO(data), spill_threshold=len(data) + 4096,
                      byte_size=len(data) + 8192)
    table = _LEAVES[fmt](holder=ms, owns_holder=False).read_arrow_table()
    assert table.num_rows == _ROWS
    assert not ms.has_spill


@pytest.mark.parametrize("fmt", list(_LEAVES))
def test_leaf_reads_over_memory_holder(fmt, payloads):
    # Regression guard — the settled in-memory path is unchanged.
    table = _LEAVES[fmt](
        holder=Memory(binary=payloads[fmt]), owns_holder=False
    ).read_arrow_table()
    assert table.num_rows == _ROWS


def test_zip_reads_over_spilled_stream():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for i in range(40):
            zf.writestr(f"m{i}.bin", os.urandom(4096))
    data = buf.getvalue()
    assert len(data) > 64 * 1024
    from yggdrasil.io.zip_file import ZipFile

    ms = _spilling(data)
    zf = ZipFile(holder=ms, owns_holder=False)
    names = zf.list_entries()
    assert len(names) == 40


# ---------------------------------------------------------------------------
# End-to-end: leaf over a real stream=True HTTPStream
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    payloads: dict[str, bytes] = {}

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_GET(self):
        body = self.payloads[self.path.split("?", 1)[0].lstrip("/")]
        head = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/octet-stream\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode()
        self.wfile.write(head + body)

    def log_message(self, *a):
        pass


class _Server(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


@pytest.fixture
def session(payloads):
    _Handler.payloads = payloads
    srv = _Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    HTTPSession._INSTANCES.clear()
    sess = HTTPSession(base_url=base)
    try:
        yield sess
    finally:
        sess.clear_connections()
        HTTPSession._INSTANCES.clear()
        srv.shutdown()


@pytest.mark.parametrize("fmt", list(_LEAVES))
def test_leaf_reads_over_stream_true_response(fmt, session):
    resp = session.get(f"/{fmt}", stream=True)
    table = _LEAVES[fmt](holder=resp.buffer, owns_holder=False).read_arrow_table()
    assert table.num_rows == _ROWS


@pytest.mark.parametrize("fmt", list(_LEAVES))
def test_streamed_table_equals_preloaded(fmt, session):
    streamed = _LEAVES[fmt](
        holder=session.get(f"/{fmt}", stream=True).buffer, owns_holder=False
    ).read_arrow_table()
    preloaded = _LEAVES[fmt](
        holder=session.get(f"/{fmt}").buffer, owns_holder=False
    ).read_arrow_table()
    assert streamed.num_rows == preloaded.num_rows == _ROWS
    assert streamed.equals(preloaded)
