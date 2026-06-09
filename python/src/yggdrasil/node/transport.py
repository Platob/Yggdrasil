"""Wire transport for yggdrasil.node: Arrow IPC stream + pickle.

Two formats move over the node API:

* **Arrow IPC stream** (``application/vnd.apache.arrow.stream``) for any
  tabular payload — pa.Table, polars.DataFrame, pandas.DataFrame, a list
  of row-dicts. Typed, columnar, zero-copy on decode, streamable, and
  spillable to disk for results that don't fit in RAM.
* **pickle** (``application/x-pickle``) for everything else — scalars,
  dicts, bytes, arbitrary Python returned by a ``@remote`` function. Uses
  cloudpickle so closures and locally-defined functions round-trip.

``serialize_result``/``deserialize_result`` pick the format from the value
shape; the streaming helpers (``iter_arrow_ipc_stream``, ``write_arrow_ipc_file``,
``iter_file_chunks``) keep peak memory near one batch on the heavy path.
"""
from __future__ import annotations

import io
import os
from typing import Any, Callable, Iterable, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc

import cloudpickle

CONTENT_TYPE_ARROW_STREAM = "application/vnd.apache.arrow.stream"
CONTENT_TYPE_PICKLE = "application/x-pickle"

_DEFAULT_CHUNK_ROWS = 65_536


# ---------------------------------------------------------------------------
# pickle
# ---------------------------------------------------------------------------

def serialize_pickle(obj: Any) -> bytes:
    return cloudpickle.dumps(obj)


def deserialize_pickle(data: bytes) -> Any:
    return cloudpickle.loads(data)


# ---------------------------------------------------------------------------
# tabular detection + coercion
# ---------------------------------------------------------------------------

def is_tabular(obj: Any) -> bool:
    if isinstance(obj, (pa.Table, pa.RecordBatch)):
        return True
    mod = type(obj).__module__.split(".", 1)[0]
    name = type(obj).__name__
    if mod == "polars" and name == "DataFrame":
        return True
    if mod == "pandas" and name == "DataFrame":
        return True
    # list[dict] of uniform row records is treated as a table.
    if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
        return True
    return False


def to_arrow_table(obj: Any) -> pa.Table:
    if isinstance(obj, pa.Table):
        return obj
    if isinstance(obj, pa.RecordBatch):
        return pa.Table.from_batches([obj])
    mod = type(obj).__module__.split(".", 1)[0]
    name = type(obj).__name__
    if mod == "polars" and name == "DataFrame":
        return obj.to_arrow()
    if mod == "pandas" and name == "DataFrame":
        return pa.Table.from_pandas(obj, preserve_index=False)
    if isinstance(obj, list):
        return pa.Table.from_pylist(obj)
    raise TypeError(
        f"to_arrow_table got {type(obj).__name__!r}, which is not tabular. "
        f"Pass a pyarrow.Table, polars/pandas DataFrame, or a list of row dicts."
    )


# ---------------------------------------------------------------------------
# Arrow IPC stream — single buffer
# ---------------------------------------------------------------------------

def write_arrow_stream_bytes(table: pa.Table) -> bytes:
    """Encode the whole table into one Arrow IPC stream buffer."""
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def write_arrow_stream(table: pa.Table, *, max_chunksize: int = _DEFAULT_CHUNK_ROWS) -> Iterator[bytes]:
    """Yield the Arrow IPC stream as length-delimited message chunks.

    Each yielded value is a slice of the encoded stream; concatenating them
    reproduces ``write_arrow_stream_bytes(table)``. Batching by row keeps the
    encoder's working set near one batch.
    """
    return iter_arrow_ipc_stream(
        iter(table.to_batches(max_chunksize=max_chunksize)), table.schema
    )


def write_arrow_stream_chunked(table: pa.Table, *, max_chunksize: int = 8192) -> Iterator[bytes]:
    return write_arrow_stream(table, max_chunksize=max_chunksize)


def iter_arrow_ipc_stream(
    batches: Iterable[pa.RecordBatch], schema: pa.Schema
) -> Iterator[bytes]:
    """Stream a valid Arrow IPC stream batch-by-batch.

    One ``ipc.new_stream`` writer accumulates into a single ``BufferOutputStream``;
    after writing each batch we snapshot only the bytes appended since the last
    yield and hand back that slice. Concatenating every yielded slice reproduces
    exactly ``write_arrow_stream_bytes`` (schema preamble, all record-batch
    messages, EOS) — so the receiver decodes it as one stream — while peak
    working memory stays near a single encoded batch, not the whole result.
    """
    # A plain BytesIO lets us read back the bytes appended since the last yield
    # without closing the stream (pa.BufferOutputStream.getvalue() would close
    # it). pyarrow wraps the Python file object via pa.output_stream.
    sink = io.BytesIO()
    out = pa.output_stream(sink)
    writer = ipc.new_stream(out, schema)
    sent = 0

    def _drain() -> bytes:
        nonlocal sent
        out.flush()
        pos = sink.tell()
        if pos <= sent:
            return b""
        sink.seek(sent)
        chunk = sink.read(pos - sent)
        sink.seek(pos)
        sent = pos
        return chunk

    # new_stream emits the schema message on open; flush it first.
    pre = _drain()
    if pre:
        yield pre
    for batch in batches:
        writer.write_batch(batch)
        chunk = _drain()
        if chunk:
            yield chunk
    writer.close()  # writes the EOS marker
    tail = _drain()
    if tail:
        yield tail


def read_arrow_stream(data: bytes) -> pa.Table:
    return ipc.open_stream(pa.py_buffer(data)).read_all()


# ---------------------------------------------------------------------------
# Arrow IPC — disk spill + zero-copy read-back
# ---------------------------------------------------------------------------

def write_arrow_ipc_file(
    path: str, batches: Iterable[pa.RecordBatch], schema: pa.Schema
) -> int:
    """Spill batches to an Arrow IPC stream file on disk; return row count."""
    rows = 0
    with pa.OSFile(path, "wb") as sink:
        with ipc.new_stream(sink, schema) as writer:
            for batch in batches:
                writer.write_batch(batch)
                rows += batch.num_rows
    return rows


def iter_file_chunks(path: str, chunk_size: int = 1 << 20) -> Iterator[bytes]:
    """Stream raw bytes off a spilled IPC file in bounded chunks."""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                return
            yield chunk


def read_arrow_ipc_file(path: str) -> pa.Table:
    """Zero-copy read of a spilled IPC stream file via memory-map."""
    return ipc.open_stream(pa.memory_map(path, "r")).read_all()


# ---------------------------------------------------------------------------
# format dispatch
# ---------------------------------------------------------------------------

def serialize_result(obj: Any) -> tuple[bytes, str]:
    """Serialise *obj*, picking Arrow stream for tabular and pickle otherwise."""
    if is_tabular(obj):
        return write_arrow_stream_bytes(to_arrow_table(obj)), CONTENT_TYPE_ARROW_STREAM
    return serialize_pickle(obj), CONTENT_TYPE_PICKLE


def deserialize_result(data: bytes, content_type: str) -> Any:
    ct = content_type.split(";", 1)[0].strip().lower()
    if ct == CONTENT_TYPE_ARROW_STREAM:
        return read_arrow_stream(data)
    if ct == CONTENT_TYPE_PICKLE:
        return deserialize_pickle(data)
    raise ValueError(
        f"deserialize_result got content-type {content_type!r}; expected "
        f"{CONTENT_TYPE_ARROW_STREAM!r} or {CONTENT_TYPE_PICKLE!r}."
    )
